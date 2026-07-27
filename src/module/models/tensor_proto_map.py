from typing import Self

import torch
from onnx import TensorProto


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
                    ["float32", "float16", "bfloat16"],
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
