from textwrap import dedent
from typing import Self, override

import torch
from applib import (
    BaseTemplate,
    Case,
    CaseConverter,
    ComboBoxOption,
    GenericConverter,
    GUIMessage,
    NumberOption,
    Option,
    UITypes,
)

from src.module.config.tas_args import TASArgs


# Defined by: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#configuration-options
class CUDATemplate(BaseTemplate):
    """ONNX Cuda Execution Provider"""

    _instance = None

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._created = False
        return cls._instance

    def __init__(self) -> None:
        if not self._created:
            super().__init__(
                name=TASArgs.cuda_ep_template_name,
                template=self._create_template(),
                icons=None,
            )
            self._created = True

    @override
    def _create_template(self) -> dict:
        return {
            "device_id": NumberOption(default=0, ui_info=GUIMessage("The device ID.")),
            "user_compute_stream": Option(
                default=str(torch.cuda.current_stream().cuda_stream),
                ui_info=GUIMessage(
                    "Defines the compute stream for the inference to run on."
                ),
            ),
            "arena_extend_strategy": ComboBoxOption(
                default="kNextPowerOfTwo",
                values={
                    "Next Power of Two": "kNextPowerOfTwo",
                    "Same as Requested": "kSameAsRequested",
                },
                ui_info=GUIMessage(
                    "The strategy for extending the device memory arena.",
                    dedent(
                        """Powers of Two can increase performance by reducing memory fragmentation
                        at the cost of higher memory usage.
                        Same as Requested reduces memory usage but can lead to poor performance due to
                        memory fragmentation and/or excessive memory allocations. 
                        """
                    ),
                ),
            ),
            "cudnn_conv_algo_search": ComboBoxOption(
                default="EXHAUSTIVE",
                converter=CaseConverter(Case.UPPER, Case.TITLE),
                ui_info=GUIMessage(
                    "The type of search done for cuDNN convolution algorithms.",
                    dedent(
                        """Exhaustive performs a complete search, which may take a while but also produces the fastest model.
                        Heuristic performs a lightweight heuristic-based search.
                        Default uses the default algorithm.
                        """
                    ),
                ),
                values=["EXHAUSTIVE", "HEURISTIC", "DEFAULT"],
            ),
            "cudnn_conv_use_max_workspace": Option(
                default="1",
                converter=GenericConverter(["0", "1"], [False, True]),
                ui_type=UITypes.SWITCH,
                ui_info=GUIMessage(
                    "Allow unrestricted workspace size",
                    dedent(
                        """This increases performance for FP16 models by allowing CuDNN to 
                        pick tensor core algorithms for convolution operations
                        (if the hardware supports tensor core operations). 
                        It might increase performance for other datatypes (FP32 etc.)
                        """
                    ),
                ),
            ),
            "cudnn_conv1d_pad_to_nc1d": Option(
                default="0",
                converter=GenericConverter(["0", "1"], [False, True]),
                ui_info=GUIMessage(
                    "Use NC1D dimension padding for CuDNN operations",
                    "May be a lot faster on some devices such as NVIDIA A100",
                ),
            ),
            "enable_cuda_graph": Option(
                default="1",
                converter=GenericConverter(["0", "1"], [False, True]),
                ui_info=GUIMessage(
                    "Enable CUDA Graphs. Provides a performance boost by reducing CPU overhead",
                    "Note that all graph nodes must use the CUDA execution provider.",
                ),
            ),
            "use_tf32": Option(
                default="1",
                converter=GenericConverter(["0", "1"], [False, True]),
                ui_info=GUIMessage(
                    "TF32 allows certain F32 matrix multiplications and convolutions to run much faster on tensor cores.",
                    "Available on NVIDIA GPUs since Ampere",
                ),
            ),
            "prefer_nhwc": Option(
                default="0",
                converter=GenericConverter(["0", "1"], [False, True]),
                ui_info=GUIMessage(
                    "The execution provider prefers NHWC operators over NCHW",
                    "Since NVIDIA tensor cores operate more efficiently with NHWC layout, enabling this option can improve performance",
                ),
            ),
        }
