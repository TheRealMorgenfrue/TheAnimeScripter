from typing import Self

from applib import ConfigBase, CoreValidationModelGenerator

from src.module.config.tas_args import TASArgs
from src.module.config.templates.onnx.execution_providers.trt_ep_template import (
    TensortRTTemplate,
)


class TensorRTConfig(ConfigBase):
    """ONNX TensorRT Execution Provider"""

    _instance = None

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._created = False
        return cls._instance

    def __init__(self) -> None:
        if not self._created:
            template = TensortRTTemplate()
            validation_model = CoreValidationModelGenerator().get_generic_model(
                model_name=template.name,
                template=template,
            )
            super().__init__(
                name=TASArgs.trt_ep_config_name,
                template=template,
                validation_model=validation_model,
                input_data=TASArgs.trt_ep_config_path,
                save_path=TASArgs.trt_ep_config_path,
            )
            self._created = True
