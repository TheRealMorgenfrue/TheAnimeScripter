from datetime import datetime
from pathlib import Path

from src.version import VERSION


class TASArgs:
    # ┌────────────────┐
    # │ TAS attributes │
    # └────────────────┘
    # General
    app_dir = Path().cwd()
    name = "The Anime Scripter Redux"
    github = "https://github.com/TheRealMorgenfrue/TheAnimeScripter"
    is_release = False

    # IO
    video_extensions = [".mp4", ".mkv", ".webm", ".avi", ".mov", ".gif"]
    image_extensions = [".png", ".jpg", ".jpeg", ".tiff", ".tif", ".exr", ".dpx"]

    # Logging
    log_dir = Path(app_dir, "logs")
    log_format = "%(asctime)s - %(module)s - %(lineno)s - %(levelname)s - %(message)s"  # %(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_use_color = True
    log_filename = datetime.now().strftime("%Y-%m-%d")
    log_disable_header = True

    # Asset directories
    assets_dir = Path(app_dir, "src", "assets")
    logo_dir = Path(assets_dir, "logos")

    # Cache paths
    cache_dir = f"{Path(app_dir, 'tas_cache')}"
    trt_cache_dir = f"{Path(cache_dir, 'trt_engines')}"
    trt_timing_cache_dir = trt_cache_dir

    # Asset paths
    tas_main_logo = f"{logo_dir.joinpath('icon.png')}"

    # Templates
    config_units = {
        "second": "seconds",
        "retry": "retries",
        "tag": "tags",
        "day": "days",
        "kB": "",
    }

    ## Main template
    main_template_name = "tas_template"
    main_themes = ["Light", "Dark", "System"]

    ## Input metadata template
    input_metadata_template_name = "input_metadata_template"

    ## ONNX Execution Provider templates
    cuda_ep_template_name = "cuda_ep_template"
    trt_ep_template_name = "trt_ep_template"
    migraphx_ep_template_name = "migraphx_ep_template"
    openvino_ep_template_name = "openvino_ep_template"

    ## ONNX IO bindings template
    io_binding_template_name = "onnx_io_binding_template"

    ## ONNX Session Options template
    onnx_session_options_template_name = "onnx_session_options_template"

    ## ONNX Run Options template
    onnx_run_options_template_name = "onnx_run_options_template"

    ## Olive template
    olive_template_name = "olive_template"

    # Configs
    config_dir = Path(app_dir, "configs")

    ## Main config
    main_config_name = "TAS"
    main_config_file = f"{main_config_name.replace(' ', '_').lower()}_config.toml"
    main_config_path = Path(config_dir, main_config_file)

    ## Olive config
    olive_config_name = "Olive"
    olive_config_file = f"{olive_config_name.replace(' ', '_').lower()}_config.toml"
    olive_config_path = Path(config_dir, olive_config_file)

    ## ONNX Execution Provider configs
    cuda_ep_config_name = "CUDA"
    cuda_ep_config_file = f"{cuda_ep_config_name}_config.toml"
    cuda_ep_config_path = Path(config_dir, cuda_ep_config_file)
    trt_ep_config_name = "TensorRT"
    trt_ep_config_file = f"{trt_ep_config_name}_config.toml"
    trt_ep_config_path = Path(config_dir, trt_ep_config_file)
    migraphx_ep_config_name = "MIGraphX"
    migraphx_ep_config_file = f"{migraphx_ep_config_name}_config.toml"
    migraphx_ep_config_path = Path(config_dir, migraphx_ep_config_file)
    openvino_ep_config_name = "OpenVINO"
    openvino_ep_config_file = f"{openvino_ep_config_name}_config.toml"
    openvino_ep_config_path = Path(config_dir, openvino_ep_config_file)

    ## ONNX Session Options config
    onnx_session_options_config_name = "ONNX Session"
    onnx_session_options_config_file = (
        f"{onnx_session_options_config_name.replace(' ', '_').lower()}_config.toml"
    )
    onnx_session_options_config_path = Path(
        config_dir, onnx_session_options_config_file
    )

    ## ONNX Run Options config
    onnx_run_options_config_name = "ONNX Run"
    onnx_run_options_config_file = (
        f"{onnx_run_options_config_name.replace(' ', '_').lower()}_config.toml"
    )
    onnx_run_options_config_path = Path(config_dir, onnx_run_options_config_file)

    # ┌────────────────────────────┐
    # │ Override AppLib attributes │
    # └────────────────────────────┘
    # General
    _core_app_name = name
    _core_app_version = VERSION
    _core_link_github = github
    _core_is_release = is_release

    # Logging
    _core_log_dir = log_dir
    _core_log_format = log_format
    _core_log_use_color = log_use_color
    _core_log_filename = log_filename
    _core_log_disable_header = False

    # Asset directories
    _core_assets_dir = assets_dir
    _core_logo_dir = logo_dir
    _core_main_logo_path = tas_main_logo

    # Templates
    _core_main_template_name = main_template_name
    _core_config_units = config_units
    _core_template_themes = main_themes

    # Configs
    _core_config_dir = config_dir
    _core_main_config_name = main_config_name
    _core_main_config_file = main_config_file
    _core_main_config_path = main_config_path
