from applib import GUIMessage, NumberOption, Option, UIGroups


def tas_scene_detection_template() -> dict:
    return {
        "scn_detect": Option(
            default=False,
            ui_group="g_scn_detect",
            ui_group_parent=[UIGroups.NESTED_CHILDREN],
            ui_info=GUIMessage("Detect scene changes"),
        ),
        "scn_detect_sens": NumberOption(
            default=50.0,
            ui_disable_self=0.0,
            min=0.0,
            max=100.0,
            ui_group="g_scn_detect",
            ui_info=GUIMessage("Scene change sensitivity"),
        ),
    }
