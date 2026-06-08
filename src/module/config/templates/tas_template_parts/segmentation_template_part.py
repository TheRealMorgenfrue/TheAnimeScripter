from applib import ComboBoxOption, CompatilityValidator, GUIMessage, Option, UIGroups

from src.module.config.runners.compatibility.encoding_compatibility import (
    compatible_bit_depth,
)


def tas_segmentation_template() -> dict:
    return {
        "segment": Option(
            default=False,
            ui_group="g_segment",
            ui_group_parent=[UIGroups.NESTED_CHILDREN],
            ui_info=GUIMessage("Segment the video (background removal)"),
            validators=[CompatilityValidator(compatible_bit_depth, ["bit_depth"])],
        ),
        "segment_method": ComboBoxOption(
            default="anime",
            ui_group="g_segment",
            values=["anime", "anime-tensorrt", "anime-directml", "cartoon"],
            ui_info=GUIMessage("Segmentation method"),
        ),
    }
