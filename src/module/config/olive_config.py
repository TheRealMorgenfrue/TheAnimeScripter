from copy import deepcopy
from typing import Self

from applib import ConfigBase, CoreValidationModelGenerator, MappingBase

from src.module.config.tas_args import TASArgs
from src.module.config.templates.onnx.olive_template import OliveTemplate


class OliveConfig(ConfigBase):
    _instance = None

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._created = False
        return cls._instance

    def __init__(self) -> None:
        if not self._created:
            template = OliveTemplate()
            validation_model = CoreValidationModelGenerator().get_generic_model(
                model_name=template.name,
                template=template,
            )
            super().__init__(
                name=TASArgs.olive_config_name,
                template=template,
                validation_model=validation_model,
                input_data=TASArgs.olive_config_path,
                save_path=TASArgs.olive_config_path,
            )
            self._created = True

    def get_workflow_config(self) -> dict:
        """Returns all settings of `self` compatible with Olive workflows."""
        output = MappingBase(deepcopy(self.get_raw()))

        if not self["to_onnx"]:
            del output["onnx_conversion"]
        if not self["peephole"]:
            del output["onnx_peephole_optimizer"]

        del output["enabled_passes"]

        return output.get_raw()
