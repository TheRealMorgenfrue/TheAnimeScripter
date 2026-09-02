from typing import override

import torch
from torch import Tensor
from torch.types import Device

from src.module.models.tensors_base import ModelTensorsBase


class VFIModelTensors(ModelTensorsBase):
    """Base class for creating and storing VFI tensors."""

    def __init__(
        self,
        width: int,
        height: int,
        batch_size: int,
    ) -> None:
        super().__init__()
        self.width = width
        self.height = height

        self.I0_IN: Tensor
        self.I1_IN: Tensor
        self.TIMESTEP_IN: Tensor
        self.PREDICTION_OUT: Tensor

        frame_shape = [batch_size, 3, self.height, self.width]
        timestep_shape = [batch_size, 1, self.height, self.width]

        self.input_names = ["img0", "img1", "timestep"]
        self.input_fill_values = [0, 0, 0.5]
        self.input_shapes = [
            frame_shape,  # I0
            frame_shape,  # I1
            timestep_shape,  # Timestep
        ]
        self.output_names = ["prediction"]
        self.output_fill_values = [0]
        self.output_shapes = [
            frame_shape,  # Output
        ]

    @override
    def create_io_tensors(
        self,
        input_types: list[torch.dtype] | torch.dtype,
        input_devices: list[Device] | Device,
        output_types: list[torch.dtype] | torch.dtype,
        output_devices: list[Device] | Device,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        input_tensors, output_tensors = super().create_io_tensors(
            input_names=self.input_names,
            input_types=input_types,
            input_fill_values=self.input_fill_values,
            input_shapes=self.input_shapes,
            input_devices=input_devices,
            output_names=self.output_names,
            output_types=output_types,
            output_fill_values=self.output_fill_values,
            output_shapes=self.output_shapes,
            output_devices=output_devices,
        )
        self.I0_IN = input_tensors[self.input_names[0]]
        self.I1_IN = input_tensors[self.input_names[1]]
        self.TIMESTEP_IN = input_tensors[self.input_names[2]]
        self.PREDICTION_OUT = output_tensors[self.output_names[0]]

        return (input_tensors, output_tensors)

    def get_dynamic_axes(
        self,
    ) -> dict[str, dict[str, str]]:
        # For torch dynamo
        # dynamic_shapes={
        #     input_names[0]: {2: Dim.DYNAMIC, 3: Dim.DYNAMIC},
        #     input_names[1]: {2: Dim.DYNAMIC, 3: Dim.DYNAMIC},
        #     input_names[2]: {2: Dim.DYNAMIC, 3: Dim.DYNAMIC},
        #     input_names[3]: {2: Dim.DYNAMIC, 3: Dim.DYNAMIC},
        # },

        # Legacy torch export script
        return {
            self.input_names[0]: {"2": "height", "3": "width"},
            self.input_names[1]: {"2": "height", "3": "width"},
            self.input_names[2]: {"2": "height", "3": "width"},
            self.output_names[0]: {"1": "height", "2": "width"},
        }
