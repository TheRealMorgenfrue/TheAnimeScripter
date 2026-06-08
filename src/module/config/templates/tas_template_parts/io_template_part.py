import os
from shutil import which

from applib import FileSelectorOption, GUIMessage, NumberOption, UITypes, validate_path


def tas_io_template() -> dict:
    return {
        "input": FileSelectorOption(
            default="",
            ui_info=GUIMessage(
                "Input video file",
            ),
        ),
        "output": FileSelectorOption(
            default=os.environ["TAS_PATH"],
            ui_info=GUIMessage(
                "Destination folder",
            ),
            ui_show_dir_only=True,
        ),
        "inpoint": NumberOption(
            default=0.0,
            min=0.0,
            max=None,
            ui_disable_self=0.0,
            ui_info=GUIMessage(
                "Input start time",
            ),
            ui_type=UITypes.SPINBOX,
        ),
        "outpoint": NumberOption(
            default=0.0,
            min=0.0,
            max=None,
            ui_disable_self=0.0,
            ui_info=GUIMessage(
                "Input end time",
            ),
            ui_type=UITypes.SPINBOX,
        ),
        "ffmpeg": FileSelectorOption(
            default=which("ffmpeg") or "",
            ui_file_filter="ffmpeg",
            ui_info=GUIMessage("Location of FFmpeg."),
            validators=validate_path,
        ),
        "ffprobe": FileSelectorOption(
            default=which("ffprobe") or "",
            ui_file_filter="ffprobe",
            ui_info=GUIMessage("Location of FFprobe."),
            validators=validate_path,
        ),
    }
