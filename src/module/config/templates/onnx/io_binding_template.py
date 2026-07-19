from typing import Self, override

from applib import (
    BaseTemplate,
    ComboBoxOption,
    Flags,
    NumberOption,
    Option,
    TextEditOption,
)
from onnx import TensorProto
from torch import Size

from src.module.config.tas_args import TASArgs


class IOBindingTemplate(BaseTemplate):
    """ONNX IO bindings"""

    _instance = None

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._created = False
        return cls._instance

    def __init__(self) -> None:
        if not self._created:
            super().__init__(
                name=TASArgs.io_binding_template_name,
                template=self._create_template(),
                icons=None,
            )
            self._created = True

    @override
    def _create_template(self) -> dict:
        return {
            "name": TextEditOption(default=""),
            "device_type": ComboBoxOption(default="cuda", values=["cpu", "cuda"]),
            "device_id": NumberOption(default=0, min=0),
            "element_type": ComboBoxOption(
                default=TensorProto.FLOAT,
                values={
                    "FP32": TensorProto.FLOAT,
                    "FP16": TensorProto.FLOAT16,
                    "BF16": TensorProto.BFLOAT16,
                },
            ),
            "shape": Option(default=None, type=Size),
            "buffer_ptr": Option(
                default=None, flags=[Flags.HIDE_IN_CLI, Flags.HIDE_IN_GUI], type=int
            ),
        }
