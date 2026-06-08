from applib import AutoTextWrap, ComboBoxOption, GUIMessage, Option, UIGroups


def tas_performance_template() -> dict:
    return {
        "precision": ComboBoxOption(
            # TODO: Implement
            default="fp16",
            values=["fp32", "fp16"],
            ui_info=GUIMessage(
                "NOT IMPLEMENTED YET! Precision for inference, default is fp16"
            ),
        ),
        "decode_method": ComboBoxOption(
            default="cpu",
            values=["cpu", "nvdec"],
            ui_info=GUIMessage(
                "Decoding backend to use, default is cpu. 'nvdec' requires an NVIDIA GPU with NVDEC support.",
            ),
        ),
        "static_trt": Option(
            default=False,
            ui_info=GUIMessage("Force Static Mode engine generation for TensorRT"),
        ),
        "compile_mode": ComboBoxOption(
            default="default",
            values=["default", "max", "max-graphs"],
            ui_info=GUIMessage(
                "[EXPERIMENTAL] Enable PyTorch compilation for CUDA models to improve performance",
                AutoTextWrap.text_format(
                    "Only compatible with CUDA workflows and may cause compatibility issues with some models. "
                    "Increases startup time and memory usage. "
                    "'default' uses standard CudaGraph workflow without compilation, "
                    "'max' uses 'max-autotune-no-cudagraphs' mode, "
                    "'max-graphs' uses 'max-autotune-no-cudagraphs' with fullGraph=True. "
                    "Both 'max' options disable CudaGraphs, which may reduce performance at lower resolutions.",
                ),
            ),
        ),
        "profile": Option(
            default=False,
            ui_info=GUIMessage(
                "Enable torch.profiler to analyze GPU/CPU performance bottlenecks"
            ),
        ),
        "benchmark": Option(
            default=False,
            ui_group="compat_bench",
            ui_group_parent=[UIGroups.DESYNC_FALSE_CHILDREN],
            ui_info=GUIMessage("Benchmark the current configuration"),
        ),
    }
