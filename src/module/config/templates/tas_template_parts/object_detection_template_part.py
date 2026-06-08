from applib import ComboBoxOption, GUIMessage, Option, UIGroups


def tas_object_detection_template() -> dict:
    return {
        "obj_detect": Option(
            default=False,
            ui_group="g_obj_detect",
            ui_group_parent=[UIGroups.NESTED_CHILDREN],
            ui_info=GUIMessage("Enable object detection"),
        ),
        "obj_detect_method": ComboBoxOption(
            default="yolov9_small-directml",
            values=[
                "yolov9_small-directml",
                "yolov9_medium-directml",
                "yolov9_large-directml",
                "yolov9_small-openvino",
                "yolov9_medium-openvino",
                "yolov9_large-openvino",
                "yolov9_small-tensorrt",
                "yolov9_medium-tensorrt",
                "yolov9_large-tensorrt",
            ],
            ui_group="g_obj_detect",
            ui_info=GUIMessage(
                "Object detection method",
            ),
        ),
        "obj_detect_disable_annotations": Option(
            default=False,
            ui_group="g_obj_detect",
            ui_info=GUIMessage(
                "Disable class labels and confidence percentages on detection boxes",
            ),
        ),
    }
