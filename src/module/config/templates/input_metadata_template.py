from typing import Self, override

from applib import (
    BaseTemplate,
    NumberOption,
    Option,
    TemplateConfig,
)

from src.module.config.tas_args import TASArgs


class InputMetadataTemplate(BaseTemplate):
    _instance = None
    __applib_template_config__ = TemplateConfig({"disable_warnings": "gui"})

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._created = False
        return cls._instance

    def __init__(self) -> None:
        if not self._created:
            super().__init__(
                name=TASArgs.input_metadata_template_name,
                template=self._create_template(),
                icons=None,
            )
            self._created = True

    @override
    def _create_template(self) -> dict:
        return {
            "Video": {
                "width": NumberOption(default=0, min=1),
                "height": NumberOption(default=0, min=1),
                "aspect_ratio": NumberOption(default=0.0, min=0.0),
                "fps": NumberOption(default=0.0, min=0.0),
                "codec": Option(default=""),
                "pixel_format": Option(default=""),
                "color_space": Option(default=""),
                "color_primaries": Option(default=""),
                "color_transfer": Option(default=""),
                "color_range": Option(default=""),
                "duration": NumberOption(default=0.0, min=0.0),
                "total_frames": NumberOption(default=0, min=0),
                "total_frames_to_process": NumberOption(default=0, min=0),
            },
        }
