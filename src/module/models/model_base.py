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

from module.models.tensor_proto_map import TensorProtoMap
from src.module.config.io_binding_config import IOBindingConfig
from src.module.config.olive_config import OliveConfig
from src.module.config.tas_config import TASConfig


class ModelBase:
    def __init__(self) -> None:
        self.logger = LoggingManager()
        self.tas_config = TASConfig()

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
        model_output = self.auto_tune_parameters(
            model_path=model_path,
            input_names=input_names,
            input_types=input_types,
            input_shapes=input_shapes,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
        self.session = self.create_inference_session(model_output)
        self.io_binding = self.apply_io_bindings(
            self.session.io_binding(),
            *self.declare_io_bindings(
                *self.get_io_tensors(  # TODO: Use model_output dtype
                    dtype=torch.float32, device=model_output.from_device()
                )
            ),
        )

    def auto_tune_parameters(
        self,
        model_path: Path,
        input_names: list[str],
        input_types: list[str],
        input_shapes: list[list[int]],
        output_names: list[str],
        dynamic_axes: dict[str, dict[str, str]],
    ) -> olive.ModelOutput:
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

        Raises
        ------
        RuntimeError
            If model optimization failed.
        ValueError
            If some model data provided by Olive is None.
        TypeError
            If input is of incorrect type.
        """
        olive_config = OliveConfig()
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

        workflow_output = olive.run(olive_config.get_workflow_config())  # type: ignore # They have outdated type hints (https://microsoft.github.io/Olive/0.12.1/reference/python_api.html)

        best_model = workflow_output.get_best_candidate()
        if best_model is not None:
            self.logger.info(f"Succesfully optimized model '{model_path}'")
            self.logger.debug(
                # AutoTextWrap().text_format(
                #     f"""
                #     Model parameters:
                #     /tModel path: {best_model.model_path}
                #     /tModel type: {best_model.model_type}
                #     /tDevice: {best_model.from_device()}
                #     /tExecution provider: {best_model.from_execution_provider()}
                #     /tMetrics: {best_model.metrics_value}
                #     """
                # )
                dedent(
                    f"""
                    Model parameters:
                    \tModel path: {best_model.model_path}
                    \tModel type: {best_model.model_type}
                    \tDevice: {best_model.from_device()}
                    \tExecution provider: {best_model.from_execution_provider()}
                    \tMetrics: {best_model.metrics_value}
                    """
                )
            )
        else:
            raise RuntimeError(f"Failed to optimize model '{model_path}'")

        return best_model

    def create_inference_session(
        self, input_model: Path | olive.ModelOutput
    ) -> onnxruntime.InferenceSession:
        """Creates a ONNX Runtime inference session for the input model.

        The session is the main entrypoint for inference.

        Parameters
        ----------
        input_model : Path | olive.ModelOutput
            The model to perform inference with.

            Can either be a path to a model file or a model optimized by Olive.

        Returns
        -------
        onnxruntime.InferenceSession
            The input model's inference session.

        Raises
        ------
        ValueError
            If some model data provided by Olive is None.
        TypeError
            If input is of incorrect type.
        """
        if isinstance(input_model, Path):
            self.logger.warning("Path input not yet supported")

            model_path = input_model
            session_options: dict[str, Any] = {}
            providers: list[str] = []
            provider_options: list[dict[str, Any]] = []
        elif isinstance(input_model, olive.ModelOutput):
            inference_config = input_model.get_inference_config()
            print(inference_config)

            model_path = input_model.model_path
            session_options = inference_config.get("session_options")  # type: ignore
            providers = inference_config.get("execution_provider")  # type: ignore
            provider_options = inference_config.get("provider_options")  # type: ignore

            errors = []
            if model_path is None:
                errors.append("model_path")
            if session_options is None:
                errors.append("session_options")
            if providers is None:
                errors.append("providers")
            if provider_options is None:
                errors.append("provider_options")
            if errors:
                err_msg = f"None detected for the following inference settings: {', '.join(errors)}"
                raise ValueError(err_msg)
        else:
            raise TypeError(
                f"Expected input to be a Path or ModelOutput, got {type(input_model).__name__}"
            )

        session = onnxruntime.InferenceSession(
            model_path,  # type: ignore
            sess_options=session_options,
            providers=providers,
            provider_options=provider_options,
        )
        session.enable_fallback()
        return session

    def apply_io_bindings(
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
                    "device_id": tensor.device.index,
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
    def get_io_tensors(
        self, dtype: torch.dtype, device: Device
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Returns the input/output tensors which should be used by the model during inference.

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

        To perform the actual inference step, call:
            ```
            self.session.run_with_iobinding(self.io_binding)
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
            The prediction(s) of the model.
        """
        ...
