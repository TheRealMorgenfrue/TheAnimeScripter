import os
from pathlib import Path
from shutil import which
from typing import Self, override

from applib import (
    AutoTextWrap,
    BaseTemplate,
    ColorPickerOption,
    ComboBoxOption,
    CompatilityValidator,
    FileSelectorOption,
    GUIMessage,
    LoggingManager,
    NumberOption,
    Option,
    TextEditOption,
    UIGroups,
    UITypes,
    change_theme,
    change_theme_color,
    validate_background,
    validate_loglevel,
    validate_path,
    validate_theme,
)

from src.module.config.runners.actions.nelux_actions import set_nelux_log_level
from src.module.config.runners.compatibility.depth_compatibility import (
    compatible_depth_model,
)
from src.module.config.runners.compatibility.encoding_compatibility import (
    compatible_bit_depth,
)
from src.module.config.runners.validators.validate_nelux import validate_nelux_loglevel
from src.module.config.runners.validators.validate_performance import validate_precision
from src.module.config.tas_args import TASArgs
from src.module.utils.types.nelux import NeluxLogLevel


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
                        "NOT IMPLEMENTED YET! Create and use a preset configuration file based on the current arguments"
                    ),
                ),
                "loglevel": ComboBoxOption(
                    default="INFO" if TASArgs.is_release else "DEBUG",
                    actions=LoggingManager().set_level,
                    ui_info=GUIMessage(f"Set log level for {TASArgs.name}"),
                    validators=validate_loglevel,
                    values=LoggingManager.LogLevel._member_names_,
                ),
                "nelux_loglevel": ComboBoxOption(
                    default="OFF" if TASArgs.is_release else "INFO",
                    actions=set_nelux_log_level,
                    ui_info=GUIMessage("Set log level for NeLux"),
                    validators=validate_nelux_loglevel,
                    values=[level.upper() for level in NeluxLogLevel._member_names_],
                ),
            },
            "Appearance": {
                "appTheme": ComboBoxOption(
                    default="System",
                    actions=change_theme,
                    ui_info=GUIMessage("Set application theme"),
                    hide_in_cli=True,
                    validators=validate_theme,
                    values=TASArgs.main_themes,
                ),
                "appColor": ColorPickerOption(
                    default="#2abdc7",
                    actions=change_theme_color,
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
            "IO": {
                "input": FileSelectorOption(
                    default="",
                    ui_info=GUIMessage(
                        "Location of the video file or the folder containing video files",
                    ),
                ),
                "output": FileSelectorOption(
                    default=f"{Path(os.environ['TAS_PATH'], 'output')}",
                    ui_info=GUIMessage(
                        "Destination folder",
                    ),
                    ui_show_dir_only=True,
                ),
                "output_format": ComboBoxOption(
                    default="mkv",
                    values=["mkv", "mp4", "webm", "mov", "avi"],
                    ui_info=GUIMessage("File output format."),
                ),
                "audio_subs": Option(
                    default=True,
                    ui_info=GUIMessage(
                        "Include audio, subtitles, and chapters in the destination file"
                    ),
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
                    ui_info=GUIMessage("Location of FFmpeg"),
                    validators=validate_path,
                ),
                "ffprobe": FileSelectorOption(
                    default=which("ffprobe") or "",
                    ui_file_filter="ffprobe",
                    ui_info=GUIMessage("Location of FFprobe"),
                    validators=validate_path,
                ),
            },
            "Performance": {
                "precision": ComboBoxOption(
                    # TODO: Implement
                    default="auto",
                    values=["FP32", "FP16", "BF16", "MXFP8", "NVFP4"],
                    ui_info=GUIMessage(
                        "NOT IMPLEMENTED YET! Precision of inference",
                        AutoTextWrap.text_format(
                            """
                            "Auto" automatically applies the lowest precision supported by your hardware.
                            
                            Lower precision is significantly faster than higher precision, while potentially reducing quality. 
                            Please test the result of using lower precisions before processing large videos.
                            """
                        ),
                    ),
                    validators=validate_precision,
                ),
                "decode_method": ComboBoxOption(
                    default="cpu",
                    values=["cpu", "nvdec"],
                    ui_info=GUIMessage(
                        "Decoding method to use",
                        '"nvdec" requires an NVIDIA GPU with NVDEC support',
                    ),
                ),
                "static_trt": Option(
                    default=False,
                    ui_info=GUIMessage(
                        "Force static mode engine generation for TensorRT"
                    ),
                ),
                "compile_mode": ComboBoxOption(
                    default="default",
                    values=["default", "max", "max-graphs"],
                    ui_info=GUIMessage(
                        "[EXPERIMENTAL] Enable PyTorch compilation for CUDA models to improve performance",
                        AutoTextWrap.text_format(
                            """
                            Only compatible with CUDA workflows and may cause compatibility issues with some models.
                            Increases startup time and memory usage.
                            "default" uses standard CudaGraph workflow without compilation,
                            "max" uses "max-autotune-no-cudagraphs" mode,
                            "max-graphs" uses "max-autotune-no-cudagraphs" with "fullGraph=True".
                            Both "max" options disable CudaGraphs, which may reduce performance at lower resolutions.
                            """
                        ),
                    ),
                ),
                "profile": Option(
                    default=False,
                    ui_info=GUIMessage(
                        "Enable the Torch profiler to analyze GPU/CPU performance bottlenecks"
                    ),
                ),
                "benchmark": Option(
                    default=False,
                    ui_group="compat_bench",
                    ui_group_parent=[UIGroups.DESYNC_FALSE_CHILDREN],
                    ui_info=GUIMessage("Benchmark the current configuration"),
                ),
            },
            "Encoding": {
                "encode_method": ComboBoxOption(
                    default="x264",
                    values=sorted(
                        [
                            "x264",
                            "slow_x264",
                            "x264_10bit",
                            "x264_animation",
                            "x264_animation_10bit",
                            "x265",
                            "slow_x265",
                            "x265_10bit",
                            "nvenc_h264",
                            "slow_nvenc_h264",
                            "nvenc_h265",
                            "slow_nvenc_h265",
                            "nvenc_h265_10bit",
                            "nvenc_av1",
                            "slow_nvenc_av1",
                            "qsv_h264",
                            "qsv_h265",
                            "qsv_h265_10bit",
                            "av1",
                            "slow_av1",
                            "h264_amf",
                            "hevc_amf",
                            "hevc_amf_10bit",
                            "prores",
                            "prores_segment",
                            "gif",
                            "vp9",
                            "qsv_vp9",
                            "lossless",
                            "lossless_nvenc",
                            "png",
                            "nvenc_h264_nelux",
                            "nvenc_h265_nelux",
                            "nvenc_av1_nelux",
                        ]
                    ),
                    ui_info=GUIMessage("Encoding method"),
                ),
                "custom_encoder": TextEditOption(
                    default="", ui_info=GUIMessage("Custom encoder settings")
                ),
                "bit_depth": ComboBoxOption(
                    default="8bit",
                    values=["8bit", "16bit"],
                    ui_info=GUIMessage("Bit depth of the raw pipe input to FFmpeg"),
                ),
            },
            "VFI": {
                "vfi": Option(
                    default=False,
                    ui_group="g_vfi",
                    ui_group_parent=[UIGroups.NESTED_CHILDREN],
                    ui_info=GUIMessage("Interpolate the video"),
                ),
                "vfi_factor": NumberOption(
                    default=2.0,
                    min=0,
                    max=None,
                    ui_group="g_vfi",
                    ui_info=GUIMessage("Interpolation factor"),
                ),
                "vfi_model": ComboBoxOption(
                    default="rife4.6",
                    values=sorted(
                        [
                            "distildrba",
                            "distildrba-lite",
                            "distildrba-tensorrt",
                            "distildrba-lite-tensorrt",
                            "atr",
                            "rife4.6",
                            "rife4.15-lite",
                            "rife4.16-lite",
                            "rife4.17",
                            "rife4.18",
                            "rife4.20",
                            "rife4.21",
                            "rife4.22",
                            "rife4.22-lite",
                            "rife4.25",
                            "rife4.25-depth",
                            "rife4.25-lite",
                            "rife4.25-heavy",
                            "rife-ncnn",
                            "rife4.6-ncnn",
                            "rife4.15-lite-ncnn",
                            "rife4.16-lite-ncnn",
                            "rife4.17-ncnn",
                            "rife4.18-ncnn",
                            "rife4.20-ncnn",
                            "rife4.21-ncnn",
                            "rife4.22-ncnn",
                            "rife4.22-lite-ncnn",
                            "rife4.6-tensorrt",
                            "rife4.15-tensorrt",
                            "rife4.17-tensorrt",
                            "rife4.18-tensorrt",
                            "rife4.20-tensorrt",
                            "rife4.21-tensorrt",
                            "rife4.22-tensorrt",
                            "rife4.22-lite-tensorrt",
                            "rife4.25-tensorrt",
                            "rife4.25-lite-tensorrt",
                            "rife4.25-heavy-tensorrt",
                            "rife-tensorrt",
                            "gmfss",
                            "gmfss-tensorrt",
                            "rife_elexor",
                            "rife_elexor-tensorrt",
                            "rife4.6-tensorrt",
                            "rife4.6-directml",
                            "rife4.15-directml",
                            "rife4.17-directml",
                            "rife4.18-directml",
                            "rife4.20-directml",
                            "rife4.21-directml",
                            "rife4.22-directml",
                            "rife4.22-lite-directml",
                            "rife4.25-directml",
                            "rife4.25-lite-directml",
                            "rife4.25-heavy-directml",
                            "rife4.6-openvino",
                            "rife4.15-openvino",
                            "rife4.17-openvino",
                            "rife4.18-openvino",
                            "rife4.20-openvino",
                            "rife4.21-openvino",
                            "rife4.22-openvino",
                            "rife4.22-lite-openvino",
                            "rife4.25-openvino",
                            "rife4.25-lite-openvino",
                            "rife4.25-heavy-openvino",
                        ]
                    ),
                    ui_group="g_vfi",
                    ui_info=GUIMessage("Interpolation method"),
                ),
                "slowmo": Option(
                    default=False,
                    ui_group="g_vfi",
                    ui_info=GUIMessage(
                        "Enable slow motion interpolation. This will slow down the video instead of increasing the frame rate"
                    ),
                ),
                "ensemble": Option(
                    default=False,
                    ui_group="g_vfi",
                    ui_info=GUIMessage("Use the ensemble model for interpolation"),
                ),
                "dynamic_scale": Option(
                    default=False,
                    ui_group="g_vfi",
                    ui_info=GUIMessage(
                        "Use dynamic scaling for interpolation. This can improve the quality of the interpolation at the cost of performance",
                        "This is experimental and only works with Rife CUDA",
                    ),
                ),
                "static_step": Option(
                    default=False,
                    ui_group="g_vfi",
                    ui_info=GUIMessage(
                        "Force static timestep generation for Rife CUDA"
                    ),
                ),
                "vfi_first": Option(
                    default=False,
                    ui_group="g_vfi",
                    ui_info=GUIMessage(
                        "Process frames with interpolation-first pipeline order",
                        "If disabled, frames are processed with interpolation-last pipeline order",
                    ),
                ),
            },
            "SR": {
                "sr": Option(
                    default=False,
                    ui_group="g_sr",
                    ui_group_parent=[UIGroups.NESTED_CHILDREN],
                    ui_info=GUIMessage("Upscale the video"),
                ),
                "sr_factor": ComboBoxOption(
                    default=2,
                    values=[2, 3, 4],
                    ui_group="g_sr",
                    ui_info=GUIMessage("Upscaling factor"),
                ),
                "sr_model": ComboBoxOption(
                    default="shufflecugan",
                    values=sorted(
                        [
                            "shufflecugan",
                            "fallin_soft",
                            "fallin_soft-tensorrt",
                            "fallin_soft-directml",
                            "fallin_strong",
                            "fallin_strong-tensorrt",
                            "fallin_strong-directml",
                            "compact",
                            "ultracompact",
                            "superultracompact",
                            "span",
                            "compact-directml",
                            "ultracompact-directml",
                            "superultracompact-directml",
                            "shufflespan-directml",
                            "span-directml",
                            "shufflecugan-ncnn",
                            "shufflecugan-directml",
                            "shufflecugan-openvino",
                            "span-ncnn",
                            "compact-tensorrt",
                            "ultracompact-tensorrt",
                            "superultracompact-tensorrt",
                            "span-tensorrt",
                            "shufflecugan-tensorrt",
                            "shufflespan-tensorrt",
                            "open-proteus",
                            "open-proteus-tensorrt",
                            "open-proteus-directml",
                            "aniscale2",
                            "aniscale2-tensorrt",
                            "aniscale2-directml",
                            "rtmosr",
                            "rtmosr-tensorrt",
                            "rtmosr-directml",
                            "saryn",
                            "saryn-tensorrt",
                            "saryn-directml",
                            "animesr",
                            "animesr-tensorrt",
                            "animesr-directml",
                            "animesr-openvino",
                            "compact-openvino",
                            "ultracompact-openvino",
                            "superultracompact-openvino",
                            "span-openvino",
                            "open-proteus-openvino",
                            "aniscale2-openvino",
                            "shufflespan-openvino",
                            "rtmosr-openvino",
                            "saryn-openvino",
                            "fallin_soft-openvino",
                            "fallin_strong-openvino",
                            "gauss",
                            "gauss-tensorrt",
                            "gauss-directml",
                            "gauss-openvino",
                        ]
                    ),
                    ui_group="g_sr",
                    ui_info=GUIMessage("Upscaling method"),
                ),
                "custom_model": FileSelectorOption(
                    default="",
                    ui_file_filter=None,
                    ui_group="g_sr",
                    ui_info=GUIMessage("Path to custom upscaling model"),
                    validators=validate_path,
                ),
            },
            "Deduplication": {
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
                        AutoTextWrap.text_format(
                            """
                            Smooth deduplication. 
                            This will remove duplicates while also generating new frames to make the video smoother.
                            """
                        ),
                        AutoTextWrap.text_format(
                            """
                            This is experimental and may not work well with all videos.
                            Use --vfi_model to set the interpolation model.
                            """
                        ),
                    ),
                ),
            },
            "Scene Detection": {
                "scn_detect": Option(
                    default=False,
                    ui_group="g_scn_detect",
                    ui_group_parent=[UIGroups.NESTED_CHILDREN],
                    ui_info=GUIMessage("Detect scene changes"),
                ),
                "scn_detect_method": ComboBoxOption(
                    default="pyscenedetect",
                    values=sorted(
                        [
                            "pyscenedetect",
                            "maxxvit-directml",
                            "maxxvit-tensorrt",
                            "transnetv2",
                        ]
                    ),
                    ui_group="g_scn_detect",
                    ui_info=GUIMessage("Scene detection method"),
                ),
                "scn_detect_sens": NumberOption(
                    default=50.0,
                    ui_disable_self=0.0,
                    min=0.0,
                    max=100.0,
                    ui_group="g_scn_detect",
                    ui_info=GUIMessage("Scene change sensitivity"),
                ),
            },
            "Object Detection": {
                "obj_detect": Option(
                    default=False,
                    ui_group="g_obj_detect",
                    ui_group_parent=[UIGroups.NESTED_CHILDREN],
                    ui_info=GUIMessage("Enable object detection"),
                ),
                "obj_detect_method": ComboBoxOption(
                    default="yolov9_small-directml",
                    values=sorted(
                        [
                            "yolov9_small-directml",
                            "yolov9_medium-directml",
                            "yolov9_large-directml",
                            "yolov9_small-openvino",
                            "yolov9_medium-openvino",
                            "yolov9_large-openvino",
                            "yolov9_small-tensorrt",
                            "yolov9_medium-tensorrt",
                            "yolov9_large-tensorrt",
                        ]
                    ),
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
            },
            "Segmentation": {
                "segment": Option(
                    default=False,
                    ui_group="g_segment",
                    ui_group_parent=[UIGroups.NESTED_CHILDREN],
                    ui_info=GUIMessage("Segment the video (background removal)"),
                    validators=[
                        CompatilityValidator(compatible_bit_depth, ["bit_depth"])
                    ],
                ),
                "segment_method": ComboBoxOption(
                    default="anime",
                    ui_group="g_segment",
                    values=["anime", "anime-tensorrt", "anime-directml", "cartoon"],
                    ui_info=GUIMessage("Segmentation method"),
                ),
            },
            "Depth Estimation": {
                "depth": Option(
                    default=False,
                    ui_group="g_depth",
                    ui_group_parent=[UIGroups.NESTED_CHILDREN],
                    ui_info=GUIMessage("Estimate the depth of the video"),
                ),
                "depth_method": ComboBoxOption(
                    default="small_v2",
                    values=sorted(
                        [
                            "small_v2",
                            "base_v2",
                            "large_v2",
                            "giant_v2",
                            "distill_small_v2",
                            "distill_base_v2",
                            "distill_large_v2",
                            "og_small_v2",
                            "og_base_v2",
                            "og_large_v2",
                            "og_giant_v2",
                            "og_distill_small_v2",
                            "og_distill_base_v2",
                            "og_distill_large_v2",
                            "og_video_small_v2",
                            "og_video_base_v2",
                            "og_video_large_v2",
                            "video_small_v2",
                            "video_large_v2",
                            "small_v2-tensorrt",
                            "base_v2-tensorrt",
                            "large_v2-tensorrt",
                            "distill_small_v2-tensorrt",
                            "distill_base_v2-tensorrt",
                            "distill_large_v2-tensorrt",
                            "small_v2-directml",
                            "base_v2-directml",
                            "large_v2-directml",
                            "distill_small_v2-directml",
                            "distill_base_v2-directml",
                            "distill_large_v2-directml",
                            "og_small_v2-tensorrt",
                            "og_base_v2-tensorrt",
                            "og_large_v2-tensorrt",
                            "og_distill_small_v2-tensorrt",
                            "og_distill_base_v2-tensorrt",
                            "og_distill_large_v2-tensorrt",
                            "small_v3",
                            "base_v3",
                            "large_v3",
                            "giant_v3",
                            "small_v3-directml",
                            "base_v3-directml",
                            "large_v3-directml",
                            "giant_v3-directml",
                            "small_v3-tensorrt",
                            "base_v3-tensorrt",
                            "large_v3-tensorrt",
                            "giant_v3-tensorrt",
                            "small_v2-openvino",
                            "base_v2-openvino",
                            "large_v2-openvino",
                            "distill_small_v2-openvino",
                            "distill_base_v2-openvino",
                            "distill_large_v2-openvino",
                            "og_small_v2-openvino",
                            "og_base_v2-openvino",
                            "og_large_v2-openvino",
                            "small_v3-openvino",
                            "base_v3-openvino",
                            "large_v3-openvino",
                            "giant_v3-openvino",
                        ]
                    ),
                    ui_group="g_depth",
                    ui_info=GUIMessage(
                        "Depth estimation method",
                    ),
                ),
                "depth_quality": ComboBoxOption(
                    default="low",
                    values=["low", "medium", "high"],
                    ui_info=GUIMessage(
                        "This will determine the quality of the depth map",
                        AutoTextWrap().text_format(
                            """
                            Low is significantly faster, but lower quality.
                            Only works with CUDA depth maps.
                            """
                        ),
                    ),
                    validators=CompatilityValidator(
                        compatible_depth_model, ["depth_method"]
                    ),
                ),
            },
        }
