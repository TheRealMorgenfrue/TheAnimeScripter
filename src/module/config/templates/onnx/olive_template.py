from pathlib import Path
from typing import Self, override

from applib import (
    AutoTextWrap,
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
                            "Pytorch Model",
                            "ONNX Model",
                            "OpenVINO Model",
                            "TensorFlow Model",
                            "QNN Model",
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
                # "io_config": {
                #     "input_names": Option(type=list[str]),
                #     "input_types": Option(type=list[str]),
                #     "input_shapes": Option(type=list[list[int]]),
                #     "output_names": Option(type=list[str]),
                #     "dynamic_axes": Option(
                #         default={
                #             "input": Option(default={"0": "batch_size"}),
                #             "output": Option(default={"0", "batch_size"}),
                #         },
                #         type=dict[str, dict[str, str]],
                #     ),
                # },
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
                    "peephole": Option(
                        default=True,
                        ui_info=GUIMessage(
                            "Optimize ONNX model by fusing nodes",
                            "Runs a combination of onnxscript optimizer, onnxoptimizer, reshape fusion, and cast chain elimination",
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
