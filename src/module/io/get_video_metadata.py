import json
import subprocess

from src.module.config.input_metadata_config import InputMetadataConfig
from src.module.config.tas_config import TASConfig
from src.module.errors import MetadataError


def get_video_metadata(input_path: str) -> InputMetadataConfig:
    """Get metadata from a video file using FFprobe.

    Parameters
    ----------
    input_path : str
        The path to the input file to gather metadata from.

    Returns
    -------
    InputMetadataConfig
        The metadata gathered by FFprobe.

    Raises
    ------
    MetadataError
        If FFprobe failed to gather metadata.
    applib.CoreValidationError
        If the metadata is invalid.
    """
    config = TASConfig()
    cmd = [
        config["ffprobe"],
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        "-count_packets",
        input_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
    except subprocess.CalledProcessError as e:
        stderr = f"\n{e.stderr}" if e.stderr else ""
        msg = f"Failed extract video metadata: {e}\n\t{e.stdout}{stderr}"
        raise MetadataError(msg) from None

    if not result.stdout:
        msg = f"No output received from ffprobe{f'\n\t{result.stderr}' if result.stderr else ''}"
        raise MetadataError(msg)

    probe_data = json.loads(result.stdout)
    video_stream: dict | None = None
    for stream in probe_data["streams"]:
        # Get video stream
        if stream["codec_type"] == "video" and video_stream is None:
            video_stream = stream

    if video_stream is None:
        raise MetadataError("No video stream found")

    # Extract metadata
    width = int(video_stream["width"])
    height = int(video_stream["height"])
    fpsParts = video_stream["r_frame_rate"].split("/")
    fps = float(fpsParts[0]) / float(fpsParts[1])
    total_frames = int(video_stream.get("nb_read_packets", 0))
    out_point = config["outpoint"]

    metadata = {
        "width": width,
        "height": height,
        "aspect_ratio": round(width / height, 2),
        "fps": round(fps, 2),
        "codec": video_stream["codec_name"],
        "pixel_format": video_stream.get("pix_fmt", "unknown"),
        "color_space": video_stream.get("color_space", "unknown"),
        "color_primaries": video_stream.get("color_primaries", "unknown"),
        "color_transfer": video_stream.get("color_transfer", "unknown"),
        "color_range": video_stream.get("color_range", "unknown"),
        "duration": float(probe_data["format"]["duration"]),
        "total_frames": total_frames,
        "total_frames_to_process": int((out_point - config["inpoint"]) * fps)
        if out_point
        else total_frames,
    }
    return InputMetadataConfig({"Video": metadata})
