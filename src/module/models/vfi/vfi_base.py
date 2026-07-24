import math
from fractions import Fraction
from pathlib import Path
from typing import override

import torch
from torch import Tensor

from src.module.config.io_binding_config import IOBindingConfig
from src.module.models.model_base import ModelBase


class VFIModelBase(ModelBase):
    """Base class for VFI models"""

    def __init__(self, model_path: Path, vfi_factor: float | int) -> None:
        super().__init__(model_path, {}, {})
        self.vfi_factor = vfi_factor
        self.frame_counter = 0
        self.warmup = True

        if isinstance(self.vfi_factor, float):
            factor = Fraction(self.vfi_factor).limit_denominator(100)
            self.vfi_factor_numerator = factor.numerator
            self.vfi_factor_denominator = factor.denominator
        else:
            self.vfi_factor_numerator = self.vfi_factor
            self.vfi_factor_denominator = 1

    def _compute_timesteps(self) -> list[float]:
        """Computes the timesteps between two frames where interpolated
        frames should be inserted.

        The algorithm works for all vfi_factor >= 1.

        Examples
        --------
        NOTE: The two given frames are denoted as frame 0 and frame 1.

        - Given `vfi_factor=2`, the timestep is `[0.5]` (the middle of frame 0 and frame 1).
        - Given `vfi_factor=2.5`, the timesteps alternates between `[0.4]` for frame 0 and `[0.2, 0.6]` for frame 1.
        - Given `vfi_factor=3`, the timesteps are `[0.33, 0.66]`.

        Returns
        -------
        list[float]
            A list of timesteps.
        """
        # TODO: This function can be computed ahead-of-time, on the GPU
        if isinstance(self.vfi_factor, float):
            current_index = self.frame_counter
            next_index = current_index + 1

            output_start = (
                current_index * self.vfi_factor_numerator
            ) // self.vfi_factor_denominator
            output_end = (
                next_index * self.vfi_factor_numerator
            ) // self.vfi_factor_denominator

            self.frame_counter += 1
            return [
                (
                    output_start
                    + i * self.vfi_factor_denominator % self.vfi_factor_numerator
                )
                / self.vfi_factor_numerator
                for i in range(1, output_end - output_start)
            ]
        else:
            return [i / self.vfi_factor for i in range(1, self.vfi_factor)]

    def cache_frame_reset(self, frame: Tensor) -> None:
        """
        Reset the temporal state with a new frame.
        Called on scene changes.
        """
        self.process_frame(frame, "I0")

    def process_frame(self, frame: Tensor, name: str) -> None: ...

    @override
    def inference(self, frame: Tensor, **kwargs) -> None:
        if self.warmup:
            self.process_frame(frame, "F0")
            self.process_frame(frame, "I0")
            self.warmup = False
            return

        self.process_frame(frame, "I1")
        for timestep in self._compute_timesteps():
            pass


class VFIModelTensorsBase:
    def __init__(
        self,
        width: int,
        height: int,
        multiplier: int,
        channels: int,
        dtype: torch.dtype,
        device: torch.Device,
    ) -> None:
        padded_width = self.compute_resolution_padding(width, multiplier)
        padded_height = self.compute_resolution_padding(height, multiplier)

        self.I0 = torch.zeros(
            1,
            3,
            padded_height,
            padded_width,
            dtype=dtype,
            device=device,
        )
        self.I1 = torch.zeros(
            1,
            3,
            padded_height,
            padded_width,
            dtype=dtype,
            device=device,
        )
        self.F0 = torch.zeros(
            1,
            channels,
            padded_height,
            padded_width,
            dtype=dtype,
            device=device,
        )

        self.F1 = torch.zeros(
            1,
            channels,
            padded_height,
            padded_width,
            dtype=dtype,
            device=device,
        )

        self.timestep = torch.full(
            (1, 1, padded_height, padded_width),
            0.5,
            dtype=dtype,
            device=device,
        )

        self.output = torch.zeros(
            (1, 3, height, width),
            device=device,
            dtype=dtype,
        )

    def create_input_tensors(
        self,
        padded_width,
        padded_height: int,
        channels: int,
        dtype: torch.dtype,
        device: torch.Device,
    ) -> list[Tensor]:
        img0_input = torch.zeros(
            1, 3, padded_height, padded_width, dtype=dtype, device=device
        )
        img1_input = torch.zeros(
            1, 3, padded_height, padded_width, dtype=dtype, device=device
        )
        timestep_input = torch.full(
            (1, 1, padded_height, padded_width),
            0.5,
            dtype=dtype,
            device=device,
        )
        f0_input = torch.zeros(
            1,
            channels,
            padded_height,
            padded_width,
            dtype=dtype,
            device=device,
        )

        inputList = [img0_input, img1_input, timestep_input, f0_input]
        inputNames = ["img0", "img1", "timestep", "f0"]
        outputNames = ["output", "f1"]
        dynamicAxes = {
            "img0": {2: "height", 3: "width"},
            "img1": {2: "height", 3: "width"},
            "timestep": {2: "height", 3: "width"},
            "output": {1: "height", 2: "width"},
            "f0": {2: "height", 3: "width"},
        }

    def compute_resolution_padding(self, frame_axis: int, multiplier: int) -> int:
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
