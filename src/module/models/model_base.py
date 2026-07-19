from abc import abstractmethod
from pathlib import Path
from textwrap import dedent
from typing import Any

import olive
import onnxruntime
from applib import LoggingManager
from torch import Tensor

from src.module.config.io_binding_config import IOBindingConfig
from src.module.config.olive_config import OliveConfig
from src.module.config.tas_config import TASConfig


class ModelBase:
    """Base class for all models"""

    def __init__(self) -> None:
        self.logger = LoggingManager()
        self.tas_config = TASConfig()

    def auto_tune_parameters(self, model_path: Path) -> olive.ModelOutput | None:
        """Automatically optimizes the model for inference on the user's hardware.

        The model is converted to ONNX format.

        Parameters
        ----------
        model_path : Path
            The path to the original model.
            May be in any format supported by Olive (check Olive's template).

        Returns
        -------
        ModelOutput | None
            The the optimized ONNX model if succesful, `None` otherwise.
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
            self.logger.error(f"Failed to optimize model '{model_path}'")

        return best_model

    def create_inference_session(
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

    @abstractmethod
    def declare_io_bindings(
        self,
    ) -> tuple[list[IOBindingConfig], list[IOBindingConfig]]:
        """Define the input and output tensors used by the model.

        Returns
        -------
        tuple[list[IOBindingConfig], list[IOBindingConfig]]
            A tuple of binding configurations, where:
            - `tuple[0]` is the list of input tensor bindings.
            - `tuple[1]` is the list of output tensor bindings.
        """
        ...

    @abstractmethod
    def __call__(self, frame: Tensor, **kwargs) -> None:
        """Performs inference using a video frame as input.

        Parameters
        ----------
        frame : Tensor
            The video frame to perform inference on.
        **kwargs : dict
            Additional arguments which might be used by the model.

            NOTE: All subclassed models must accept keyword arguments even if not used!
        """
        ...
