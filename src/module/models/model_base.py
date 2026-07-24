from abc import abstractmethod
from pathlib import Path
from textwrap import dedent
from typing import Any

import olive
import onnxruntime
import torch
from applib import LoggingManager
from torch import Tensor

from src.module.config.io_binding_config import IOBindingConfig
from src.module.config.olive_config import OliveConfig
from src.module.config.tas_config import TASConfig


class ModelBase:
    """Base class for all models"""

    def __init__(
        self, model_path: Path, inputs: dict[str, Tensor], outputs: dict[str, Tensor]
    ) -> None:
        """Initialize a model for inference.

        Parameters
        ----------
        model_path : Path
            The location of the model to initialize.
        input_names : list[str]
            The names of the arguments in the model's forward pass.

            For instance, consider the following model's forward pass:
            ```
            def forward(self, img0, img1, timestep, f0): ...
            ```
            Here, the `input_names` is: `["img0", "img1", "timestep", "f0"]`.

        output_names : list[str]
            The names of the returned values in the model's forward pass.

            For instance, consider the following return values:
            ```
            return (warped_img0 * mask + warped_img1 * (1 - mask))[
                :, :, : self.height, : self.width
            ], f1
            ```
            Here, the `output_names` would be: `["output", "f1"]`.

        Raises
        ------
        RuntimeError
            If model optimization failed.
        ValueError
            If some model data provided by Olive is None.
        TypeError
            If input is of incorrect type.
        """
        self.logger = LoggingManager()
        self.tas_config = TASConfig()

        if self.tas_config["auto_tune"]:
            input_model = self._auto_tune_parameters(model_path)  # RuntimeError
        else:
            input_model = model_path  # TODO: Convert to ONNX if not one already

        # The session controls inference and is main entrypoint for inference.
        self.session = self._create_inference_session(input_model)
        # The IO binding controls input/output tensors for inference and defines dataflow.
        self.io_binding = self._apply_io_bindings(
            self.session.io_binding(), *self._declare_io_bindings(input_model)
        )

    def _auto_tune_parameters(self, model_path: Path) -> olive.ModelOutput:
        """Automatically optimizes the model for inference on the user's hardware.

        The model is converted to ONNX format.

        Parameters
        ----------
        model_path : Path
            The path to the original model.
            May be in any format supported by Olive (check Olive's template).

        Returns
        -------
        ModelOutput
            The the optimized ONNX model.

        Raises
        ------
        RuntimeError
            If model optimization failed.
        """
        olive_config = OliveConfig()
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

    def _create_inference_session(
        self, input_model: Path | olive.ModelOutput
    ) -> onnxruntime.InferenceSession:
        """Creates a ONNX Runtime inference session for the input model.

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

    def _apply_io_bindings(
        self,
        io_binding: onnxruntime.IOBinding,
        input_bindings: list[IOBindingConfig],
        output_bindings: list[IOBindingConfig],
    ) -> onnxruntime.IOBinding:
        """Apply the IO binding configurations to the ONNX Runtime IOBinding.

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
        return io_binding

    def _declare_io_bindings(
        self,
        # inputs: dict[str, Tensor],
        # outputs: dict[str, Tensor],
        test: olive.ModelOutput,
    ) -> tuple[list[IOBindingConfig], list[IOBindingConfig]]:
        """Define the input and output tensors used by the model.

        Parameters
        ----------
        input_names : list[str]
            The names of the arguments in the model's forward pass.

            For instance, consider the following model's forward pass:
            ```
            def forward(self, img0, img1, timestep, f0): ...
            ```
            Here, the `input_names` is: `["img0", "img1", "timestep", "f0"]`.

        output_names : list[str]
            The names of the returned values in the model's forward pass.

            For instance, consider the following return values:
            ```
            return (warped_img0 * mask + warped_img1 * (1 - mask))[
                :, :, : self.height, : self.width
            ], f1
            ```
            Here, the `output_names` would be: `["output", "f1"]`.

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
        # for d in [inputs, None, outputs]:
        #     if d is None:
        #         is_output = True
        #         continue

        #     for name, tensor in d.items():
        #         data = {
        #             "name": name,
        #             "device_type": "cuda",  # TODO: Make a setting
        #             "device_id": 0,  # TODO: Auto-select based on device_type setting
        #             "element_type": self.tas_config["precision"],
        #             "shape": tensor.shape,
        #             "buffer_ptr": tensor.data_ptr(),
        #         }

        #         if is_output:
        #             output_configs.append(IOBindingConfig(f"{name}_outbinding", data))
        #         else:
        #             input_configs.append(IOBindingConfig(f"{name}_inbinding", data))

        conf = test.get_inference_config()
        print(conf)

        return (input_configs, output_configs)

    @abstractmethod
    def inference(self, frame: Tensor, **kwargs) -> list[Tensor]:
        """Performs inference using a video frame as input.

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
