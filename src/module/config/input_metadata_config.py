from typing import Self

from applib import ConfigBase, CoreValidationModelGenerator

from src.module.config.templates.input_metadata_template import InputMetadataTemplate


class InputMetadataConfig(ConfigBase):
    _instance = None

    def __new__(cls, metadata: dict) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._created = False
        return cls._instance

    def __init__(self, metadata: dict) -> None:
        if not self._created:
            template = InputMetadataTemplate()
            validation_model = CoreValidationModelGenerator().get_generic_model(
                model_name=template.name,
                template=template,
            )
            super().__init__(
                name="input_metadata",  # Defined here as this config is in-memory only
                template=template,
                validation_model=validation_model,
                input_data=metadata,
                save_path=None,
            )
            self._created = True
