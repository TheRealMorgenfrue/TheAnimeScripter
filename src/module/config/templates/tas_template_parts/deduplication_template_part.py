from applib import GUIMessage, NumberOption, Option, UIGroups


def tas_deduplication_template() -> dict:
    return {
        "dedup": Option(
            default=False,
            ui_group="g_dedup",
            ui_group_parent=[UIGroups.NESTED_CHILDREN],
            ui_info=GUIMessage("Deduplicate the video"),
        ),
        "dedup_sens": NumberOption(
            default=35.0,
            min=0.1,
            max=100.0,
            ui_group="g_dedup",
            ui_info=GUIMessage("Deduplication sensitivity"),
        ),
        "smooth_dedup": Option(
            default=False,
            ui_group="g_dedup",
            ui_info=GUIMessage(
                "Smooth deduplication, this will remove duplicates while also generating new frames to "
                "make the video smoother, this is experimental and may not work well with all videos, "
                "use --interpolate_method to set the interpolation method"
            ),
        ),
    }
