from pathlib import Path
from textwrap import dedent
from typing import Self, override

from applib import (
    BaseTemplate,
    ComboBoxOption,
    FileSelectorOption,
    Flags,
    GenericConverter,
    GUIMessage,
    NumberOption,
    Option,
    TextEditOption,
    UIGroups,
)

from src.module.config.tas_args import TASArgs


# Olive settings from: https://microsoft.github.io/Olive/reference/options.html#input-model-information
class OliveTemplate(BaseTemplate):
    _instance = None

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._created = False
        return cls._instance

    def __init__(self) -> None:
        if not self._created:
            super().__init__(
                name=TASArgs.olive_template_name,
                template=self._create_template(),
                icons=None,
            )
            self._created = True

    @override
    def _create_template(self) -> dict:
        return {
            "workflow_id": TextEditOption(
                default="",
                ui_info=GUIMessage(
                    "The workflow ID to use for the current config",
                    "If left blank, an ID is automatically generated",
                ),
            ),
            "input_model": {
                "type": ComboBoxOption(
                    default="PytorchModel",
                    converter=GenericConverter(
                        [
                            "PytorchModel",
                            "ONNXModel",
                            "OpenVINOModel",
                            "TensorFlowModel",
                            "QNNModel",
                        ],
                        [
                            "Pytorch",
                            "ONNX",
                            "OpenVINO",
                            "TensorFlow",
                            "QNN",
                        ],
                    ),
                    ui_group_parent=[UIGroups.DISABLE_CHILDREN],
                    ui_group="model_type",
                    ui_info=GUIMessage("The model type to optimize"),
                    values=[
                        "PytorchModel",
                        "ONNXModel",
                        "OpenVINOModel",
                        "TensorFlowModel",
                        "QNNModel",
                    ],
                ),
                "model_path": FileSelectorOption(
                    default="",
                    ui_info=GUIMessage(
                        "The path to the model file",
                    ),
                ),
                "model_script": FileSelectorOption(
                    default="", ui_file_filter="Python (*.py)", ui_disable_button=False
                ),
                "model_loader": TextEditOption(
                    default="",
                    ui_info=GUIMessage(
                        "Enter the name of the function that loads the model.",
                        "It should take the model_path as an argument and return the loaded PyTorch model.",
                    ),
                ),
                "io_config": {
                    "input_names": Option(default=[], type=list[str]),
                    "input_types": Option(default=[], type=list[str]),
                    "input_shapes": Option(default=[], type=list[list[int]]),
                    "output_names": Option(default=[], type=list[str]),
                    "dynamic_axes": Option(
                        default={},
                        type=dict[str, dict[str, str]],
                    ),
                },
            },
            "systems": {"local_system": {"type": "LocalSystem"}},
            "evaluators": "common_evaluator",
            # {
            # "common_evaluator": {
            #     "metrics": [
            #         {
            #             "name": "latency",
            #             "type": "latency",
            #             "sub_types": [
            #                 {
            #                     "name": "avg",
            #                     "goal": {
            #                         "type": "percent-min-improvement",
            #                         "value": 20,
            #                     },
            #                 },
            #                 {"name": "max"},
            #                 {"name": "min"},
            #             ],
            #             "user_config": {
            #                 "user_script": "user_script.py",
            #                 "data_dir": "data",
            #                 "dataloader_func": "create_dataloader",
            #                 "batch_size": 16,
            #             },
            #         }
            #     ]
            # }
            # },
            "passes": {
                "enabled_passes": {
                    "to_onnx": Option(
                        default=True,
                        ui_info=GUIMessage(
                            "Convert the model to ONNX",
                            "This allows for aggressive model optimizations",
                        ),
                    ),
                    "peephole": Option(
                        default=True,
                        ui_info=GUIMessage(
                            "Optimize a ONNX model by fusing nodes",
                            "Runs a combination of onnxscript optimizer, onnxoptimizer, reshape fusion, and cast chain elimination",
                        ),
                    ),
                },
                "onnx_conversion": {
                    "type": Option(
                        default="OnnxConversion",
                        flags=[Flags.HIDE_IN_CLI, Flags.HIDE_IN_GUI],
                    ),
                    "user_script": Option(
                        default=None,
                        type=Path | str,
                        ui_info=GUIMessage(
                            "Path to user script",
                            "The values for other parameters which were assigned function or object names will be imported from this script.",
                        ),
                    ),
                    "script_dir": Option(
                        default=None,
                        type=Path | str,
                        ui_info=GUIMessage(
                            "Directory containing user script dependencies"
                        ),
                    ),
                    "save_as_external_data": Option(
                        default=False,
                        ui_info=GUIMessage(
                            "Serializes tensor data to separate files instead of directly in the ONNX file",
                            "Large models (>2GB) may be forced to save external data regardless of the value of this parameter",
                        ),
                    ),
                    "all_tensors_to_one_file": Option(
                        default=True,
                        ui_info=GUIMessage(
                            dedent(
                                """
                                Effective only if save_as_external_data is True. 
                                If true, save all tensors to one external file specified by "external_data_name". 
                                If false, save each tensor to a file named with the tensor name.
                                """
                            )
                        ),
                    ),
                    "external_data_name": Option(
                        default=None,
                        type=str,
                        ui_info=GUIMessage(
                            dedent(
                                """
                                Effective only if all_tensors_to_one_file is True and save_as_external_data is True. 
                                If not specified, the external data file will be named with <model_path_name>.data
                                """
                            )
                        ),
                    ),
                    "size_threshold": NumberOption(
                        default=1024,
                        min=0,
                        ui_info=GUIMessage(
                            "Effective only if save_as_external_data is True. Threshold for size of data.",
                            dedent(
                                """
                                Only when tensor's data is >= the size_threshold it will be converted to external data. 
                                To convert every tensor with raw data to external data set size_threshold=0
                                """
                            ),
                        ),
                    ),
                    "convert_attribute": Option(
                        default=False,
                        ui_info=GUIMessage(
                            dedent(
                                """
                                Effective only if save_as_external_data is True. 
                                If true, convert all tensors to external data. 
                                If false, convert only non-attribute tensors to external data
                                """
                            )
                        ),
                    ),
                    "target_opset": NumberOption(
                        default=20,
                        min=1,
                        ui_info=GUIMessage(
                            "The version of the default (ai.onnx) opset to target"
                        ),
                    ),
                    "use_dynamo_exporter": Option(
                        default=False,
                        ui_info=GUIMessage(
                            "Whether to use dynamo_export API to export ONNX model."
                        ),
                    ),
                    "past_key_value_name": TextEditOption(
                        default="past_key_values",
                        ui_info=GUIMessage(
                            "The arguments name to point to past key values",
                            dedent(
                                """
                                For model loaded from huggingface, it is 'past_key_values'. 
                                Basically, it is used only when use_dynamo_exporter is True.
                                """
                            ),
                        ),
                    ),
                    "device": TextEditOption(
                        default="cuda",
                        ui_info=GUIMessage(
                            "The device to use for model conversion",
                            'If not specified, will use "cpu" for PyTorch model and "cuda" for DistributedHfModel',
                        ),
                    ),
                    "dtype": ComboBoxOption(
                        default=None,
                        values=["float32", "float16", "bfloat16"],
                        ui_info=GUIMessage(
                            "The dtype to cast the model to before conversion",
                            "If not specified, will use the model as is",
                        ),
                    ),
                    "parallel_jobs": Option(
                        default=None,
                        min=0,
                        type=int,
                        ui_info=GUIMessage(
                            "Number of parallel jobs",
                            "Defaulted to number of CPUs. Set it to 0 to disable",
                        ),
                    ),
                    "merge_adapter_weights": Option(
                        default=False,
                        ui_info=GUIMessage(
                            "Whether to merge adapter weights before conversion",
                            dedent(
                                """
                                After merging, the model structure is consistent with base model. 
                                That is useful if you cannot run conversion for some fine-tuned models 
                                with adapter weights
                                """
                            ),
                        ),
                    ),
                    "save_metadata_for_token_generation": Option(
                        default=False,
                        ui_info=GUIMessage(
                            "Whether to save metadata for token generation",
                            "Includes config.json, generation_config.json, and tokenizer related files",
                        ),
                    ),
                    "optimize": Option(
                        default=True,
                        ui_info=GUIMessage(
                            "Whether to optimize the model by exporting with constant folding and redundancies elimination"
                        ),
                    ),
                    "dynamic": Option(
                        default=True,
                        ui_info=GUIMessage(
                            "Whether to export the model with dynamic axes/shapes",
                            "Do not change this setting unless you know what you're doing",
                        ),
                    ),
                },
                "onnx_peephole_optimizer": {
                    "type": Option(
                        default="OnnxPeepholeOptimizer",
                        flags=[Flags.HIDE_IN_CLI, Flags.HIDE_IN_GUI],
                    ),
                    "onnxscript_optimize": Option(
                        default=True,
                        ui_info=GUIMessage(
                            "Run onnxscript optimizer for general graph optimizations"
                        ),
                    ),
                    "onnxoptimizer_optimize": Option(
                        default=True,
                        ui_info=GUIMessage(
                            "Run onnxoptimizer for additional graph optimizations"
                        ),
                    ),
                    "fuse_reshape_operations": Option(
                        default=True,
                        ui_info=GUIMessage(
                            "Fuse consecutive Reshape operators where the latter flattens to [-1]"
                        ),
                    ),
                    "cast_chain_elimination": Option(
                        default=True,
                        ui_info=GUIMessage(
                            "Apply a targeted rewrite rule to eliminate redundant round-trip cast chains",
                            "For instance, fp32 → fp16 → fp32 → identity produced by dynamo export",
                        ),
                    ),
                },
            },
            "engine": {
                # "search_strategy": {
                #     "execution_order": ComboBoxOption(
                #         default="joint",
                #         ui_info=GUIMessage(
                #             "The execution order of the optimizations of passes"
                #         ),
                #         values=["pass-by-pass", "joint"],
                #     ),
                #     "sampler": ComboBoxOption(
                #         default="sequential",
                #         ui_info=GUIMessage(
                #             "The search sampler to use while traversing",
                #             AutoTextWrap.text_format(
                #                 """
                #                 Sampler details:
                #                 random: Samples random points from the search space.
                #                 sequential: Iterates over the entire search space sequentially.
                #                 tpe: Sample using TPE (Tree-structured Parzen Estimator) algorithm.
                #                 """
                #             ),
                #         ),
                #         values=["random", "sequential", "tpe"],
                #     ),
                #     "max_time": NumberOption(
                #         default=120,
                #         min=1,
                #         max=None,
                #         ui_info=GUIMessage(
                #             "The maximum time of the search in seconds",
                #             "Only valid for joint execution order",
                #         ),
                #     ),
                # },
                "host": "local_system",
                "target": "local_system",
                "cache_dir": FileSelectorOption(
                    default=f"{Path(TASArgs.app_dir, 'olive_cache')}",
                    ui_show_dir_only=True,
                    ui_info=GUIMessage("Folder to store cache data in"),
                ),
                "output_dir": FileSelectorOption(
                    default=f"{Path(TASArgs.app_dir, 'olive_model_optim')}"
                ),
                "log_severity_level": ComboBoxOption(
                    default=2 if TASArgs.is_release else 0,
                    ui_info=GUIMessage("The log level of Olive"),
                    values={
                        "DEBUG": 0,
                        "INFO": 1,
                        "WARNING": 2,
                        "ERROR": 3,
                        "CRITICAL": 4,
                    },
                ),
                "ort_log_severity_level": ComboBoxOption(
                    default=2 if TASArgs.is_release else 0,
                    ui_info=GUIMessage("The log level of ONNX Runtime C++ logs"),
                    values={
                        "DEBUG": 0,
                        "INFO": 1,
                        "WARNING": 2,
                        "ERROR": 3,
                        "CRITICAL": 4,
                    },
                ),
                "ort_py_log_severity_level": ComboBoxOption(
                    default=2 if TASArgs.is_release else 0,
                    ui_info=GUIMessage("The log level of ONNX Runtime Python logs"),
                    values={
                        "DEBUG": 0,
                        "INFO": 1,
                        "WARNING": 2,
                        "ERROR": 3,
                        "CRITICAL": 4,
                    },
                ),
                "log_to_file": Option(
                    default=TASArgs.is_release,
                    ui_info=GUIMessage("This decides whether to write logs to a file"),
                ),
            },
        }
