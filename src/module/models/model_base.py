import os
import re
import shutil
from abc import abstractmethod
from pathlib import Path
from textwrap import dedent
from typing import Any

import olive
import onnxruntime
import torch
from applib import LoggingManager
from torch import Tensor
from torch.types import Device

from src.module.config.olive_config import OliveConfig
from src.module.config.onnx_config.cuda_ep_config import CUDAConfig
from src.module.config.onnx_config.io_binding_config import IOBindingConfig
from src.module.config.onnx_config.run_option_config import RunOptionsConfig
from src.module.config.onnx_config.session_option_config import SessionOptionConfig
from src.module.config.onnx_config.tensorrt_ep_config import TensorRTConfig
from src.module.config.tas_config import TASConfig
from src.module.utils.hardware_checkers.hardware_checker import HardwareChecker
from src.module.utils.type_mappings import (
    ExecutionProviderMap,
    ORTSessionOptionMap,
    TensorProtoMap,
)


class ModelBase:
    def __init__(self) -> None:
        self.logger = LoggingManager()
        self.tas_config = TASConfig()
        self.sessions: list[onnxruntime.InferenceSession] = []
        self.io_bindings: list[onnxruntime.IOBinding] = []

        run_options = RunOptionsConfig()
        self.run_options = onnxruntime.RunOptions()
        for k, v, _ in run_options:
            self.run_options.add_run_config_entry(k, v)

    def prepare_model(
        self,
        model_path: Path,
        input_names: list[str],
        input_types: list[str],
        input_shapes: list[list[int]],
        output_names: list[str],
        dynamic_axes: dict[str, dict[str, str]],
    ):
        """Prepare the selected model for inference.

        This involves:

        1. Loading the model.
        2. Converting it to ONNX (if needed).
        3. Optimizing it (if requested).
        4. Creating input/output model tensors and binding them to a ONNX Runtime.

        Parameters
        ----------
        model_path : Path
            The location of the model to initialize.
        input_names : list[str]
            The names the arguments in the model's forward pass.
            See `ModelTensorsBase.create_io_tensors` for a description.
        input_types : list[str]
            The data types of the tensors defined by `input_names`.
        input_shapes : list[list[int]]
            The shapes of the tensors defined by `input_names`.
        output_names : list[str]
            The names of the returned values in the model's forward pass.
            See `ModelTensorsBase.create_io_tensors` for a description.
        dynamic_axes : dict[str, dict[str, str]]
            The dynamic axes of the model.
            See `ModelTensorsBase.get_dynamic_axes` for a description.
        """
        optimized_model_path = self.get_optimized_model_path(
            models_dir=self.tas_config["model_dir"],
            model_path=f"{model_path}",
            model_extension=os.path.splitext(os.path.split(model_path)[1])[1],
        )
        if optimized_model_path.exists():
            model_output = optimized_model_path
        elif self.tas_config["autotune"]:
            model_output = self.auto_tune_parameters(
                model_path=model_path,
                input_names=input_names,
                input_types=input_types,
                input_shapes=input_shapes,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )
        else:
            extension = os.path.splitext(os.path.split(model_path)[1])[1]
            match extension:
                case ".onnx":
                    pass
                case ".pth", "pt":
                    # TODO: Implement
                    raise NotImplementedError("Work in progress")
                case _:
                    raise NotImplementedError(
                        f"Model type '{model_path}' is not supported"
                    )
            model_output = model_path

        session_options = self.ort_get_session_configuration()
        execution_providers = HardwareChecker().get_execution_providers("all")
        for device_name in self.tas_config["devices"]:
            device_id = HardwareChecker().get_device_id(device_name)
            if device_id is None:
                self.logger.warning(
                    f"Failed to find device ID for device: {device_name}"
                )
                continue

            providers = execution_providers[device_id]
            provider_options = self.ort_get_provider_configs(providers)

            # NOTE: ONNX Runtime only supports one GPU per session: https://github.com/microsoft/onnxruntime/issues/16382#issuecomment-1594933698
            # Multi-GPU inference needs multiple sessions with either:
            # - Round-robin input cycling (frame batch 0 -> GPU 0, frame batch 1 -> GPU 1 ...)
            # - For tasks with multiple videos, each GPU can process their own video (episode 1 -> GPU 0, episode 2 -> GPU 1)
            session = self.ort_create_inference_session(
                model_path=model_output,
                session_options=session_options,
                providers=providers,
                provider_options=provider_options,
            )
            io_binding = self.ort_apply_io_bindings(
                session.io_binding(),
                *self.declare_io_bindings(
                    *self.declare_io_tensors(
                        dtype=TensorProtoMap().get_torch(  # TODO: Make precision device independent
                            self.tas_config["precision"]
                        ),
                        device=ExecutionProviderMap().get_torch(  # REVIEW: Possible deadlock if provider[0] is not assigned to IO tensors?
                            session.get_providers()[0]
                        ),
                    )
                ),
            )
            self.sessions.append(session)
            self.io_bindings.append(io_binding)

    def auto_tune_parameters(
        self,
        model_path: Path,
        input_names: list[str],
        input_types: list[str],
        input_shapes: list[list[int]],
        output_names: list[str],
        dynamic_axes: dict[str, dict[str, str]],
    ) -> Path:
        """Initialize a model for inference.

        Parameters
        ----------
        model_path : Path
            The location of the model to initialize.
        input_names : list[str]
            The names the arguments in the model's forward pass.
            See `ModelTensorsBase.create_io_tensors` for a description.
        output_names : list[str]
            The names of the returned values in the model's forward pass.
            See `ModelTensorsBase.create_io_tensors` for a description.
        dynamic_axes : dict[str, dict[str, str]]
            The dynamic axes of the model.
            See `ModelTensorsBase.get_dynamic_axes` for a description.

        Returns
        -------
        Path
            The path to the optimized onnx model.

        Raises
        ------
        RuntimeError
            If model optimization failed.
        """
        extension = os.path.splitext(os.path.split(model_path)[1])[1]
        is_not_onnx = "onnx" not in extension
        olive_config = OliveConfig()
        olive_config.set_value("to_onnx", is_not_onnx, path="enabled_passes")
        if is_not_onnx:
            olive_config.set_value("input_names", input_names, path="io_config")
            olive_config.set_value(
                "input_types",
                input_types,
                path="io_config",
            )
            olive_config.set_value(
                "input_shapes",
                input_shapes,
                path="io_config",
            )
            olive_config.set_value("output_names", output_names, path="io_config")
            olive_config.set_value("dynamic_axes", dynamic_axes, path="io_config")
        else:
            del olive_config["io_config"]
            del olive_config["model_script"]
            del olive_config["model_loader"]

        workflow_output = olive.run(olive_config.get_workflow_config())  # type: ignore # They have outdated type hints (https://microsoft.github.io/Olive/0.12.1/reference/python_api.html)

        best_model = workflow_output.get_best_candidate()
        if (
            best_model is not None
            and best_model.model_path is not None
            and best_model.model_type is not None
        ):
            model_extension = os.path.splitext(os.path.split(best_model.model_path)[1])[
                1
            ]
            optimized_model_path = self.get_optimized_model_path(
                models_dir=self.tas_config["model_dir"],
                model_path=f"{model_path}",
                model_extension=model_extension,
            )
            shutil.copyfile(
                best_model.model_path,
                optimized_model_path,
            )
            self.logger.info(f"Succesfully optimized model '{model_path}'", pid=0)
            self.logger.debug(
                dedent(
                    f""" 
                    Model parameters:
                    \tModel path: {best_model.model_path}
                    \tModel type: {best_model.model_type}
                    \tDevice: {best_model.from_device()}
                    \tExecution provider: {best_model.from_execution_provider()}
                    \tMetrics: {best_model.metrics_value}
                    """
                ),
                pid=0,
            )
        else:
            raise RuntimeError(f"Failed to optimize model '{model_path}'")

        # # Parse model output
        # inference_config = best_model.get_inference_config()
        # session_options = inference_config.get("session_options")
        # providers = inference_config.get("execution_provider")
        # provider_options = inference_config.get("provider_options")

        # if session_options is None:
        #     self.logger.warning(
        #         'Olive produced no "session_options". Using config instead', pid=0
        #     )
        #     session_options = self.ort_get_session_configuration()

        # if providers is None:
        #     self.logger.warning(
        #         'Olive produced no "providers". Using config instead', pid=0
        #     )
        #     providers = HardwareChecker().get_execution_providers(
        #         self.tas_config["devices"]
        #     )  # TODO: Make it handle multiple GPUs
        # if provider_options is None:
        #     self.logger.warning(
        #         'Olive produced no "provider_options". Using config instead', pid=0
        #     )
        #     provider_options = self.ort_get_provider_configs(providers)

        return optimized_model_path

    @staticmethod
    def get_optimized_model_path(
        models_dir: str, model_path: str, model_extension: str
    ) -> Path:
        model_name = os.path.splitext(os.path.split(model_path)[1])[0]

        type_match = re.search(rf".*[/\\]?{models_dir}[/\\](.*)[/\\].*", model_path)
        if type_match:
            model_type = type_match.group(0)
            return Path(
                models_dir,
                model_type,
                "optim",
                f"{model_name}{model_extension}",
            )
        else:
            raise RuntimeError(
                f"Failed to infer model type of {model_path!r} (got {type_match!r})"
            )

    @staticmethod
    def ort_get_provider_configs(providers: list[str]) -> list[dict[str, Any]]:
        """Returns the configs of the `providers`.

        Parameters
        ----------
        providers : list[str]
            The execution providers to get configs of.

        Returns
        -------
        list[dict[str, Any]]
            A list of execution provider configs corresponding to `providers`.
        """
        configs = []
        for provider in providers:
            match provider:
                case "CPUExecutionProvider":
                    configs.append({})
                case "CUDAExecutionProvider":
                    configs.append(CUDAConfig().get_raw())
                case "TensorrtExecutionProvider":
                    configs.append(TensorRTConfig().get_raw())
                case "OpenVINOExecutionProvider":
                    raise NotImplementedError()
                case "MIGraphXExecutionProvider":
                    raise NotImplementedError()
        return configs

    @staticmethod
    def ort_get_session_configuration() -> onnxruntime.SessionOptions:
        session_options = onnxruntime.SessionOptions()
        sess_conf = SessionOptionConfig()
        for k, v, _ in sess_conf:
            try:
                v = f"{ORTSessionOptionMap().get_ort(v)}"
            except KeyError:
                pass
            session_options.add_session_config_entry(k, v)
        # session_options.graph_optimization_level = (
        #     onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        # )
        return session_options

    def ort_create_inference_session(
        self,
        model_path: Path | str,
        session_options: onnxruntime.SessionOptions,
        providers: list[str],
        provider_options: list[dict[str, Any]],
    ) -> onnxruntime.InferenceSession:
        """Creates a ONNX Runtime inference session for the input model.

        The session is the main entrypoint for inference.

        Parameters
        ----------
        model_path : Path | str
            The model to perform inference with.
        session_options : onnxruntime.SessionOptions
            Settings for the inference session.

            Please see SessionOptionsTemplate for details.
        providers : list[str]
            The execution providers to use for this session.

            See: https://onnxruntime.ai/docs/execution-providers/
        provider_options : list[dict[str, Any]]
            Settings for the providers. For instance, which device to execute on.

            Please see the provider templates for details.

        Returns
        -------
        onnxruntime.InferenceSession
            An inference session of the model tied to a particular device.
        """
        self.logger.debug("Starting inference session", pid=0)
        # onnxruntime.set_default_logger_severity(0)
        session = onnxruntime.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=providers,
            provider_options=provider_options,
        )
        session.enable_fallback()
        return session

    def ort_apply_io_bindings(
        self,
        io_binding: onnxruntime.IOBinding,
        input_bindings: list[IOBindingConfig],
        output_bindings: list[IOBindingConfig],
    ) -> onnxruntime.IOBinding:
        """Apply the IO binding configurations to the ONNX Runtime IOBinding.

        The IO binding controls input/output tensors for inference and defines dataflow.

        Parameters
        ----------
        io_binding : onnxruntime.IOBinding
            The ONNX Runtime IOBinding.
        input_bindings : list[IOBindingConfig]
            A list of input binding configurations.
        output_bindings : list[IOBindingConfig]
            A list of output binding configurations.

        Returns
        -------
        onnxruntime.IOBinding
            The ONNX Runtime IOBinding mapped to the IO configurations.
        """
        for ib in input_bindings:
            io_binding.bind_input(**ib.get_raw())
        for ob in output_bindings:
            io_binding.bind_output(**ob.get_raw())

        self.logger.debug("Applied IO bindings", pid=0)
        return io_binding

    def declare_io_bindings(
        self,
        inputs: dict[str, Tensor],
        outputs: dict[str, Tensor],
    ) -> tuple[list[IOBindingConfig], list[IOBindingConfig]]:
        """Define the input and output tensors used by the model.

        Parameters
        ----------
        input_names : list[str]
            The names of the arguments in the model's forward pass.
        output_names : list[str]
            The names of the returned values in the model's forward pass.

        Returns
        -------
        tuple[list[IOBindingConfig], list[IOBindingConfig]]
            A tuple of binding configurations, where:
            - `tuple[0]` is the list of input tensor bindings.
            - `tuple[1]` is the list of output tensor bindings.
        """
        is_output = False
        input_configs = []
        output_configs = []
        proto_map = TensorProtoMap()
        for d in [inputs, None, outputs]:
            if d is None:
                is_output = True
                continue

            for name, tensor in d.items():
                data = {
                    "name": name,
                    "device_type": tensor.device.type,
                    "device_id": tensor.device.index
                    if tensor.device.index is not None
                    else 0,
                    "element_type": proto_map.get_proto(tensor.dtype),
                    "shape": tuple(tensor.shape),
                    "buffer_ptr": tensor.data_ptr(),
                }
                if is_output:
                    output_configs.append(IOBindingConfig(f"{name}_outbinding", data))
                else:
                    input_configs.append(IOBindingConfig(f"{name}_inbinding", data))

        return (input_configs, output_configs)

    @abstractmethod
    def declare_io_tensors(
        self, dtype: torch.dtype, device: Device
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Returns a new instance of the input/output tensors which should be used by the model during inference.

        To make it easier to define, use a subclass of `ModelTensorsBase`, as defined
        in `self.get_tensor_constructor`.

        Parameters
        ----------
        dtype : torch.dtype
            The data type of the tensors.
        device : Device
            The device to put the tensors on.

        Returns
        -------
        tuple[dict[str, Tensor],dict[str, Tensor]]
            Returns a tuple, where:
                - tuple[0] are the input tensors.
                - tuple[1] are the output tensors.
        """
        ...

    @abstractmethod
    def inference(self, frame: Tensor, **kwargs) -> list[Tensor]:
        """Performs inference using a video frame as input.

        To perform the actual inference step with ONNX Runtime (ORT), call:
            ```
            self.session.run_with_iobinding(
                self.io_binding, run_options=self.run_options
            )
            ```

        Parameters
        ----------
        frame : Tensor
            The video frame to perform inference on.
        **kwargs : dict
            Additional arguments which might be used by the model.

            NOTE: All subclassed models must accept keyword arguments even if not used!

        Returns
        -------
        list[Tensor]
            The frame prediction(s) of the model.
        """
        ...
