from typing import Self

import onnxruntime
import torch
from onnx import TensorProto
from torch.types import Device


class TensorProtoMap:
    _instance = None

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._created = False
        return cls._instance

    def __init__(self) -> None:
        """Converts between `torch.dtype`, `str`, and `onnx.TensorProto`."""
        if not self._created:
            self._torch2proto = {
                torch.float32: TensorProto.FLOAT,
                torch.float16: TensorProto.FLOAT16,
                torch.bfloat16: TensorProto.BFLOAT16,
            }
            self._proto2torch = {value: key for key, value in self._torch2proto.items()}
            self._torch2str = dict(
                zip(
                    self._torch2proto.keys(),
                    ["fp32", "fp16", "bf16"],
                    strict=True,
                )
            )
            self._str2torch = {value: key for key, value in self._torch2str.items()}
            self._created = True

    def get_torch(self, value: TensorProto.DataType | str) -> torch.dtype:
        """Returns dtype version of `value`. Used by Torch."""
        return (
            self._str2torch[value]
            if isinstance(value, str)
            else self._proto2torch[value]
        )

    def get_proto(self, value: torch.dtype) -> TensorProto.DataType:
        """Returns TensorProto version of `value`. Used by ONNX."""
        return self._torch2proto[value]

    def get_torch_str(self, value: torch.dtype) -> str:
        """Returns stringified version of `value`. Used by Olive."""
        return self._torch2str[value]


class ExecutionProviderMap:
    _instance = None

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._created = False
        return cls._instance

    def __init__(self) -> None:
        """Converts ONNX Execution Providers to torch.Device."""
        if not self._created:
            self._onnx2torch = {
                "CPUExecutionProvider": "cpu",
                "CUDAExecutionProvider": "cuda",
                "TensorrtExecutionProvider": "cuda",
                # "MIGraphXExecutionProvider": None,
                # "ROCMExecutionProvider": None,
                "OpenVINOExecutionProvider": "xpu",
                "CoreMLExecutionProvider": "mps",
            }
            # Torch
            # cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, maia, xla, lazy, vulkan, mps, meta, hpu, mtia

            # ONNX
            # 'DnnlExecutionProvider', 'TvmExecutionProvider', 'VitisAIExecutionProvider', 'QNNExecutionProvider', 'NnapiExecutionProvider', 'VSINPUExecutionProvider',
            # 'JsExecutionProvider', '', 'ArmNNExecutionProvider', 'ACLExecutionProvider', 'DmlExecutionProvider', 'RknpuExecutionProvider',
            # 'WebNNExecutionProvider', 'WebGpuExecutionProvider', 'XnnpackExecutionProvider', 'CANNExecutionProvider'
            self._created = True

    def get_torch(self, value: str) -> Device:
        """Returns the torch device corresponding to `value`."""
        return self._onnx2torch[value]


class ORTSessionOptionMap:
    _instance = None

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._created = False
        return cls._instance

    def __init__(self) -> None:
        """Converts `str` to ONNX Session Options"""
        if not self._created:
            self._str2graphopt = {
                "Disabled": onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL,
                "Light": onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC,
                "Extended": onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
                "Full": onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL,
            }
            self._created = True

    def get_ort(self, value: str) -> Device:
        """Returns the ORT Session Option corresponding to `value`."""
        return self._str2graphopt[value]
