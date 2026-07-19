import os
from pathlib import Path
from shutil import which
from typing import Self, override

from applib import (
    AutoTextWrap,
    BaseTemplate,
    CheckListOption,
    ColorPickerOption,
    ComboBoxOption,
    FileSelectorOption,
    Flags,
    GenericConverter,
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
                    flags=Flags.HIDE_IN_CLI,
                    validators=validate_theme,
                    values=TASArgs.main_themes,
                ),
                "appColor": ColorPickerOption(
                    default="#2abdc7",
                    actions=change_theme_color,
                    flags=Flags.HIDE_IN_CLI,
                    ui_info=GUIMessage("Set application color"),
                ),
                "appBackground": FileSelectorOption(
                    default="",
                    flags=Flags.HIDE_IN_CLI,
                    ui_file_filter="Images (*.jpg *.jpeg *.png *.bmp)",
                    ui_info=GUIMessage("Select background image"),
                    validators=[validate_path, validate_background],
                ),
                "backgroundOpacity": NumberOption(
                    default=50,
                    min=0,
                    max=100,
                    flags=Flags.HIDE_IN_CLI,
                    ui_info=GUIMessage(
                        "Set background opacity",
                        "A greater opacity yields a brighter background",
                    ),
                ),
                "backgroundBlur": NumberOption(
                    default=0,
                    min=0,
                    max=30,
                    flags=Flags.HIDE_IN_CLI,
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
                "decode_method": ComboBoxOption(
                    default="cpu",
                    values=["cpu", "nvdec"],
                    ui_info=GUIMessage(
                        "Decoding method to use",
                        '"nvdec" requires an NVIDIA GPU with NVDEC support',
                    ),
                ),
                "auto_tune": Option(
                    default=True,
                    ui_info=GUIMessage(
                        "Employ automatic model optimization tailored to your hardware",
                        "It may take a while to process",
                    ),
                ),
                "precision": ComboBoxOption(
                    # TODO: Implement
                    default="Auto",
                    values=["Auto", "FP32", "TF32", "FP16", "BF16", "MXFP8", "NVFP4"],
                    ui_info=GUIMessage(
                        "NOT IMPLEMENTED YET! Precision of inference",
                        AutoTextWrap.text_format(
                            """
                            "Auto" automatically applies the best precision supported by your hardware. 
                            The best precision is chosen to balance performance and model quality.
                            
                            Lower precision is significantly faster than higher precision, while potentially reducing quality. 
                            Please test the result of using lower precisions before processing large videos.
                            """
                        ),
                    ),
                    validators=validate_precision,
                ),
                "execution_providers": CheckListOption(
                    default=["CPUExecutionProvider"],
                    converter=GenericConverter(
                        [
                            "CPUExecutionProvider",
                            "CUDAExecutionProvider",
                            "TensorrtExecutionProvider",
                            "OpenVINOExecutionProvider",
                            "MIGraphXExecutionProvider",
                        ],
                        ["CPU", "CUDA", "TensorRT", "OpenVINO", "MIGraphX"],
                    ),
                    values=[
                        "CPUExecutionProvider",
                        "CUDAExecutionProvider",
                        "TensorrtExecutionProvider",
                        "OpenVINOExecutionProvider",
                        "MIGraphXExecutionProvider",
                    ],
                ),
                "static_trt": Option(
                    default=False,
                    ui_info=GUIMessage(
                        "Force static mode engine generation for TensorRT"
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
                            "rife4.6",
                            "gmfss",
                            "rife_elexor",
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
        }
