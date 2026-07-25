from fractions import Fraction
from pathlib import Path
from typing import override

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.types import Device

from src.module.models.model_base import ModelBase
from src.module.models.vfi.vfi_tensors_base import VFIModelTensors


class VFIModelBase(ModelBase):
    """Base class for VFI models"""

    def __init__(
        self,
        model_path: Path,
        width: int,
        height: int,
        vfi_factor: float | int,
        multiplier: int = 32,
        channels: int = 8,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.vfi_factor = vfi_factor
        self.frame_counter = 0
        self.warmup = True
        self.run_options = {"disable_synchronize_execution_providers": "1"}
        self.tensors = VFIModelTensors(width, height, multiplier, channels)

        if isinstance(self.vfi_factor, float):
            factor = Fraction(self.vfi_factor).limit_denominator(100)
            self.vfi_factor_numerator = factor.numerator
            self.vfi_factor_denominator = factor.denominator

        self.prepare_model(
            model_path=model_path,
            input_names=self.tensors.input_names,
            input_types=[f"{dtype}" for _ in range(len(self.tensors.input_names))],
            input_shapes=self.tensors.input_shapes,
            output_names=self.tensors.output_names,
            dynamic_axes=self.tensors.get_dynamic_axes(),
        )

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

    def process_frame(self, frame: Tensor, name: str) -> None:
        # TODO: Check if frame padding is necessary
        match name:
            case "I0":
                self.tensors.I0_IN.copy_(
                    F.pad(frame, self.tensors.padding), non_blocking=True
                )
            case "I1":
                self.tensors.I1_IN.copy_(
                    F.pad(frame, self.tensors.padding), non_blocking=True
                )
            case "F0":
                # TODO: The norm should be performed in the model (it's the head in the model, i.e. the encode step)
                self.tensors.F0_IN.copy_(
                    F.pad(frame, self.tensors.padding), non_blocking=True
                )
            case "cache_I0":
                self.tensors.I0_IN.copy_(self.tensors.I1_IN, non_blocking=True)
            case "cache_F0":
                self.tensors.F0_IN.copy_(self.tensors.F1_OUT, non_blocking=True)

    @override
    def get_io_tensors(
        self, dtype: torch.dtype, device: Device
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        return self.tensors.create_io_tensors(
            input_types=dtype,
            input_devices=device,
            output_types=dtype,
            output_devices=device,
        )

    @override
    def inference(self, frame: Tensor, **kwargs) -> list[Tensor]:
        # TODO: This function can be faster with overlapping IO and compute
        # https://onnxruntime.ai/docs/performance/device-tensor.html
        # https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#performance-tuning
        # TODO: This function can be faster with batched inference

        if self.warmup:
            self.process_frame(frame, "F0")
            self.process_frame(frame, "I0")
            self.warmup = False
            return []
        self.process_frame(frame, "I1")

        predictions = []
        for timestep in self._compute_timesteps():
            self.tensors.TIMESTEP_IN.fill_(timestep)
            self.session.run_with_iobinding(
                self.io_binding  # , run_options=self.run_options
            )  # Synchronous by default
            predictions.append(self.tensors.OUTPUT_OUT.clone())

        self.process_frame(None, "cache_I0")  # type: ignore
        self.process_frame(None, "cache_F0")  # type: ignore
        return predictions
