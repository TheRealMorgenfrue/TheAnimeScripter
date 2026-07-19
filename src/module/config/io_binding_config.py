from applib import ConfigBase, CoreValidationModelGenerator

from src.module.config.templates.onnx.io_binding_template import IOBindingTemplate


class IOBindingConfig(ConfigBase):
    def __init__(self, config_name: str, data: dict) -> None:
        """Creates an instance of a ONNX IO binding

        NOTE: This config does not follow the singleton pattern!
        """

        template = IOBindingTemplate()
        validation_model = CoreValidationModelGenerator().get_generic_model(
            model_name=template.name,
            template=template,
        )
        super().__init__(
            name=config_name,
            template=template,
            validation_model=validation_model,
            input_data=data,
            save_path=None,
        )
