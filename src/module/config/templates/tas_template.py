from typing import Self, override

import nelux
from applib import (
    BaseTemplate,
    ColorPickerOption,
    ComboBoxOption,
    FileSelectorOption,
    GUIMessage,
    LoggingManager,
    NumberOption,
    Option,
    change_theme,
    change_theme_color,
    validate_background,
    validate_loglevel,
    validate_path,
    validate_theme,
)

from module.config.runners.actions.nelux_actions import set_nelux_log_level
from module.config.runners.validators.validate_nelux import validate_nelux_loglevel
from src.module.config.tas_args import TASArgs
from src.module.config.templates.tas_template_parts.deduplication_template_part import (
    tas_deduplication_template,
)
from src.module.config.templates.tas_template_parts.depth_estimation_template_part import (
    tas_depth_estimation_template,
)
from src.module.config.templates.tas_template_parts.encoding_template_part import (
    tas_encoding_template,
)
from src.module.config.templates.tas_template_parts.io_template_part import (
    tas_io_template,
)
from src.module.config.templates.tas_template_parts.object_detection_template_part import (
    tas_object_detection_template,
)
from src.module.config.templates.tas_template_parts.performance_template_part import (
    tas_performance_template,
)
from src.module.config.templates.tas_template_parts.scene_detection_template_part import (
    tas_scene_detection_template,
)
from src.module.config.templates.tas_template_parts.segmentation_template_part import (
    tas_segmentation_template,
)
from src.module.config.templates.tas_template_parts.sr_template_part import (
    tas_sr_template,
)
from src.module.config.templates.tas_template_parts.vfi_template_part import (
    tas_vfi_template,
)


class TASTemplate(BaseTemplate):
    _instance = None

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._created = False
        return cls._instance

    def __init__(self) -> None:
        if not self._created:
            super().__init__(
                name=TASArgs.main_template_name,
                template=self._create_template(),
                icons=None,
            )
            self._created = True

    @override
    def _create_template(self) -> dict:
        return {
            "General": {
                "preset": ComboBoxOption(
                    # TODO: Implement
                    default="",
                    values=[""],
                    ui_info=GUIMessage(
                        "TODO: Implement. Create and use a preset configuration file based on the current arguments"
                    ),
                ),
                "loglevel": ComboBoxOption(
                    default="INFO" if TASArgs.is_release else "DEBUG",
                    actions=[LoggingManager().set_level],
                    ui_info=GUIMessage(f"Set log level for {TASArgs.name}"),
                    validators=[validate_loglevel],
                    values=LoggingManager.LogLevel._member_names_,
                ),
                "nelux_loglevel": ComboBoxOption(
                    default="OFF" if TASArgs.is_release else "INFO",
                    actions=[set_nelux_log_level],
                    ui_info=GUIMessage("Set log level for NeLux"),
                    validators=[validate_nelux_loglevel],
                    values=[level.upper() for level in nelux.LogLevel._member_names_],
                ),
            },
            "Appearance": {
                "appTheme": ComboBoxOption(
                    default="System",
                    actions=[change_theme],
                    ui_info=GUIMessage("Set application theme"),
                    hide_in_cli=True,
                    validators=[validate_theme],
                    values=TASArgs.main_themes,
                ),
                "appColor": ColorPickerOption(
                    default="#2abdc7",
                    actions=[change_theme_color],
                    hide_in_cli=True,
                    ui_info=GUIMessage("Set application color"),
                ),
                "appBackground": FileSelectorOption(
                    default="",
                    hide_in_cli=True,
                    ui_file_filter="Images (*.jpg *.jpeg *.png *.bmp)",
                    ui_info=GUIMessage("Select background image"),
                    validators=[validate_path, validate_background],
                ),
                "backgroundOpacity": NumberOption(
                    default=50,
                    min=0,
                    max=100,
                    hide_in_cli=True,
                    ui_info=GUIMessage(
                        "Set background opacity",
                        "A greater opacity yields a brighter background",
                    ),
                ),
                "backgroundBlur": NumberOption(
                    default=0,
                    min=0,
                    max=30,
                    hide_in_cli=True,
                    ui_info=GUIMessage(
                        "Set background blur radius",
                        "A greater radius increases the blur effect",
                    ),
                ),
            },
            "IO": tas_io_template(),
            "Performance": tas_performance_template(),
            "Encoding": tas_encoding_template(),
            "VFI": tas_vfi_template(),
            "SR": tas_sr_template(),
            "Deduplication": tas_deduplication_template(),
            "Scene Detection": tas_scene_detection_template(),
            "Object Detection": tas_object_detection_template(),
            "Segmentation": tas_segmentation_template(),
            "Depth Estimation": tas_depth_estimation_template(),
        }
