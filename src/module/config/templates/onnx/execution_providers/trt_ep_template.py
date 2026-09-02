from textwrap import dedent
from typing import Self, override

from applib import (
    BaseTemplate,
    FileSelectorOption,
    Flags,
    GUIMessage,
    NumberOption,
    Option,
    TextEditOption,
    UIGroups,
    UITypes,
)

from src.module.config.tas_args import TASArgs


class TensortRTTemplate(BaseTemplate):
    _instance = None

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._created = False
        return cls._instance

    def __init__(self) -> None:
        if not self._created:
            super().__init__(
                name=TASArgs.trt_ep_template_name,
                template=self._create_template(),
                icons=None,
            )
            self._created = True

    @override
    def _create_template(self) -> dict:
        return {
            "device_id": NumberOption(0, min=0, ui_info=GUIMessage("GPU device ID")),
            # Managed by TAS
            "user_compute_stream": Option(
                "",
                flags=[Flags.HIDE_IN_CLI, Flags.HIDE_IN_GUI],
                ui_info=GUIMessage(
                    "define the compute stream for the inference to run on. It implicitly sets the has_user_compute_stream option"
                ),
            ),
            "trt_max_workspace_size": NumberOption(
                default=1073741824,
                min=1,
                ui_info=GUIMessage(
                    "The maximum workspace size, in bytes, for the TensorRT engine"
                ),
            ),
            "trt_max_partition_iterations": NumberOption(
                default=1000,
                min=1,
                ui_info=GUIMessage(
                    "The maximum number of iterations allowed in model partitioning for TensorRT",
                    dedent(
                        """
                        If target model can't be successfully partitioned when the maximum number of iterations is reached, 
                        the whole model will fall back to other execution providers such as CUDA or CPU
                        """
                    ),
                ),
            ),
            "trt_min_subgraph_size": NumberOption(
                default=1,
                min=0,
                ui_info=GUIMessage(
                    "The minimum node size in a subgraph after partitioning",
                    "Subgraphs with smaller size will fall back to other execution providers",
                ),
            ),
            # Managed by TAS
            "trt_fp16_enable": Option(
                default=False,
                flags=[Flags.HIDE_IN_CLI, Flags.HIDE_IN_GUI],
                ui_info=GUIMessage("Enable FP16 mode in TensorRT"),
            ),
            # Managed by TAS
            # "trt_int8_enable": Option(
            #     default=False,
            #     flags=[Flags.HIDE_IN_CLI, Flags.HIDE_IN_GUI],
            #     ui_info=GUIMessage("Enable INT8 mode in TensorRT"),
            # ),
            # "trt_int8_calibration_table_name": FileSelectorOption(
            #     default="",
            #     ui_info=GUIMessage(
            #         "Specify the INT8 calibration table file for non-QDQ models in INT8 mode"
            #     ),
            # ),
            # "trt_int8_use_native_calibration_table": Option(
            #     default=False,
            #     ui_info=GUIMessage(
            #         "Select what calibration table is used for non-QDQ models in INT8 mode",
            #         dedent(
            #             """
            #             If True, native TensorRT generated calibration table is used.
            #             If False, ONNXRUNTIME tool generated calibration table is used.
            #             """
            #         ),
            #     ),
            # ),
            "trt_dla_enable": Option(
                default=True, ui_info=GUIMessage("Enable Deep Learning Accelerator")
            ),
            "trt_dla_core": NumberOption(
                default=0, min=0, ui_info=GUIMessage("Specify DLA core to execute on")
            ),
            "trt_engine_cache_enable": Option(
                default=True,
                ui_info=GUIMessage(
                    "Enable TensorRT engine caching",
                    "This makes subsequent engine build times up to 42x faster",
                ),
            ),
            "trt_engine_cache_path": FileSelectorOption(
                default=TASArgs.trt_cache_dir,
                ui_info=GUIMessage(
                    "Path for TensorRT engine and profile files",
                    "It is also the path for the INT8 calibration table file",
                ),
            ),
            "trt_engine_cache_prefix": TextEditOption(
                default="",
                ui_info=GUIMessage(
                    "Custom engine cache prefix",
                    "If this option is empty, a new engine cache with a default prefix will be generated",
                ),
            ),
            "trt_dump_subgraphs": Option(
                default=False,
                ui_info=GUIMessage(
                    "Dumps the subgraphs that are transformed into TRT engines in onnx format to the filesystem",
                    "This can help debugging subgraphs, e.g. by using 'trtexec --onnx my_model.onnx' and check the outputs of the parser",
                ),
            ),
            "trt_force_sequential_engine_build": Option(
                default=False,
                ui_info=GUIMessage(
                    "Sequentially build TensorRT engines across provider instances in multi-GPU environment"
                ),
            ),
            "trt_context_memory_sharing_enable": Option(
                default=True,
                ui_info=GUIMessage(
                    "Share execution context memory between TensorRT subgraphs"
                ),
            ),
            "trt_layer_norm_fp32_fallback": Option(
                default=False,
                ui_info=GUIMessage("Force Pow + Reduce ops in layer norm to FP32"),
            ),
            "trt_timing_cache_enable": Option(
                default=True,
                ui_info=GUIMessage(
                    "Enable the TensorRT timing cache",
                    "This makes subsequent engine build times up to 10x faster",
                ),
            ),
            "trt_timing_cache_path": FileSelectorOption(
                default=TASArgs.trt_timing_cache_dir,
                ui_info=GUIMessage("Path for TensorRT timing cache"),
            ),
            "trt_force_timing_cache": Option(
                default=False,
                ui_info=GUIMessage(
                    "Force the TensorRT timing cache to be used even if device profile does not match",
                    "A perfect match is only the exact same GPU model as the on that produced the timing cache",
                ),
            ),
            "trt_detailed_build_log": Option(
                default=False,
                ui_info=GUIMessage(
                    "Enable detailed build step logging on TensorRT EP with timing for each engine build"
                ),
            ),
            "trt_build_heuristics_enable": Option(
                default=False,
                ui_info=GUIMessage(
                    "Build engine using heuristics to reduce build time"
                ),
            ),
            "trt_cuda_graph_enable": Option(
                default=True,
                ui_info=GUIMessage(
                    "Capture a CUDA graph which can drastically help for a network with many small layers as it reduces launch overhead on the CPU"
                ),
            ),
            "trt_sparsity_enable": Option(
                default=True, ui_info=GUIMessage("Enable sparsity in TensorRT")
            ),
            "trt_builder_optimization_level": NumberOption(
                default=5,
                min=0,
                max=5,
                ui_type=UITypes.SLIDER,
                ui_info=GUIMessage(
                    "TensorRT builder optimization level",
                    "Levels below 3 do not guarantee good engine performance, but greatly improve build time",
                ),
            ),
            "trt_auxiliary_streams": NumberOption(
                default=-1,
                min=-1,
                ui_info=GUIMessage(
                    "Set maximum number of auxiliary streams per inference stream",
                    dedent(
                        """
                        This may improve performance at the cost of increased memory usage.
                        
                        Setting this value to -1 will use heuristics to determine the optimal value.
                        Setting this value to 0 will lead to optimal memory usage.
                        """
                    ),
                ),
            ),
            "trt_extra_plugin_lib_paths": TextEditOption(
                default="",
                ui_info=GUIMessage(
                    "Specify extra TensorRT plugin library paths",
                    'For instance: "libvit_plugin.so;libvit_int8_plugin.so"',
                ),
            ),
            "trt_engine_hw_compatible": Option(
                default=False,
                ui_info=GUIMessage(
                    "Enable Ampere+ hardware compatibility",
                    "Hardware-compatible engines can be reused across all Ampere+ GPU environments but may have lower throughput and/or higher latency",
                ),
            ),
        }
