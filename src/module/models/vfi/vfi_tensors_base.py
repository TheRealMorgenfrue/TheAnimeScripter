import math
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
        multiplier: int,
        channels: int,
    ) -> None:
        super().__init__()
        self.width = width
        self.height = height
        self.multiplier = multiplier
        self.channels = channels
        self.padded_width = self.compute_resolution_padding(width, multiplier)
        self.padded_height = self.compute_resolution_padding(height, multiplier)
        self.padding = (0, self.padded_width - width, 0, self.padded_height - height)

        self.I0_IN: Tensor
        self.I1_IN: Tensor
        self.TIMESTEP_IN: Tensor
        self.F0_IN: Tensor
        self.OUTPUT_OUT: Tensor
        self.F1_OUT: Tensor

        frame_shape = [1, 3, self.padded_height, self.padded_width]
        timestep_shape = [1, 1, self.padded_height, self.padded_width]
        encoded_frame_shape = [1, self.channels, self.padded_height, self.padded_width]

        self.input_names = ["img0", "img1", "timestep", "f0"]
        self.input_shapes = [
            frame_shape,  # I0
            frame_shape,  # I1
            timestep_shape,  # Timestep
            encoded_frame_shape,  # F0
        ]
        self.output_names = ["output", "f1"]
        self.output_shapes = [
            frame_shape,  # Output
            encoded_frame_shape,  # F1
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
            input_shapes=self.input_shapes,
            input_devices=input_devices,
            output_names=self.output_names,
            output_types=output_types,
            output_shapes=self.output_shapes,
            output_devices=output_devices,
        )

        self.I0_IN = input_tensors[self.input_names[0]]
        self.I1_IN = input_tensors[self.input_names[1]]
        self.TIMESTEP_IN = input_tensors[self.input_names[2]]
        self.F0_IN = input_tensors[self.input_names[3]]
        self.OUTPUT_OUT = output_tensors[self.output_names[0]]
        self.F1_OUT = output_tensors[self.output_names[1]]

        return (input_tensors, output_tensors)

    def get_dynamic_axes(
        self,
    ) -> dict[str, dict[str, str]]:
        return {
            self.input_names[0]: {"2": "height", "3": "width"},
            self.input_names[1]: {"2": "height", "3": "width"},
            self.input_names[2]: {"2": "height", "3": "width"},
            self.input_names[3]: {"2": "height", "3": "width"},
            self.output_names[0]: {"1": "height", "2": "width"},
        }

    @staticmethod
    def compute_resolution_padding(frame_axis: int, multiplier: int) -> int:
        """Returns the frame size, e.g., height, with padding.

        Parameters
        ----------
        frame_size : int
            The size of one axis of the frame in pixels, e.g. 720.
        multiplier : int
            The model multiplier, e.g. 64.

        Returns
        -------
        int
            The padded frame size, e.g. 768.
        """
        return math.ceil(frame_axis / multiplier) * multiplier
