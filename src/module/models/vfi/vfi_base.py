from fractions import Fraction
from pathlib import Path
from typing import override

import torch
from torch import Tensor
from torch.types import Device

from module.utils.hardware_checkers.hardware_checker import HardwareChecker
from src.module.models.model_base import ModelBase
from src.module.models.vfi.testing_grounds.vfi_model import VFIModel
from src.module.models.vfi.vfi_tensors_base import VFIModelTensors
from src.module.models.vfi.vfi_utils import (
    compute_timesteps,
)


class VFIModelBase(ModelBase):
    def __init__(
        self,
        model_path: Path,
        width: int,
        height: int,
        vfi_factor: float | int,
        scale: float,
        batch_size: int,
        total_frames: int,
        device: Device,
        dtype: torch.dtype,
    ) -> None:
        """Base class for VFI models.

        Parameters
        ----------
        model_path : Path
            The location of the model to initialize.
        vfi_factor : float | int
            Interpolation factor.
            For instance, 2 would result in: 24 FPS -> 48 FPS.
        tensors : VFIModelTensors
            The tensors used by the model.
        dtype : torch.dtype
            The precision of the input tensors.
        """
        super().__init__()
        self.logger.debug(f"Initializing VFI model on {device} with {dtype}", pid=0)
        self.width = width
        self.height = height
        self.vfi_factor = vfi_factor
        self.scale = scale
        self.batch_size = batch_size
        self.total_frames = total_frames
        self.frame_counter = -1
        self.frame_batch_counter = 0
        self.devices = [
            HardwareChecker().get_device_id(device_name)
            for device_name in self.tas_config["devices"]
        ]
        self.device_count = len(self.devices)
        self.tensors: dict[int, VFIModelTensors] = {}

        # self.prepare_model(
        #     model_path=model_path,
        #     input_names=self.tensors.input_names,
        #     input_types=[
        #         TensorProtoMap().get_torch_str(dtype)
        #         for _ in range(len(self.tensors.input_names))
        #     ],
        #     input_shapes=self.tensors.input_shapes,
        #     output_names=self.tensors.output_names,
        #     dynamic_axes=self.tensors.get_dynamic_axes(),
        # )
        self.timesteps: list[list[float]] = []
        self.is_fraction = isinstance(self.vfi_factor, float)

        # self.vfi_model = VFIModel(
        #     batch_size=self.batch_size,
        #     width=self.tensors.width,
        #     height=self.tensors.height,
        #     vfi_factor=vfi_factor,
        #     scale=scale,
        #     dtype=dtype,
        #     device=device,
        # )
        self.img0_frame_buffer: list[Tensor] = []
        self.img1_frame_buffer: list[Tensor] = []

    def cache_frame_reset(self, frame: Tensor) -> None:
        """
        Reset the temporal state with a new frame.
        Called on scene changes.
        """
        self.process_frame(frame, "I0")

    def process_frame(self, frame: Tensor, name: str) -> None:
        match name:
            case "I0":
                self.tensors.I0_IN.copy_(frame, non_blocking=True)
            case "I1":
                self.tensors.I1_IN.copy_(frame, non_blocking=True)
            case "cache_I0":
                self.tensors.I0_IN.copy_(self.tensors.I1_IN, non_blocking=True)

    @override
    def declare_io_tensors(
        self, dtype: torch.dtype, device_id: int
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        self.logger.debug(
            f"Creating io tensors of {dtype} on {HardwareChecker().get_device_name(device_id)}",
            pid=0,
        )
        tensors = VFIModelTensors(
            width=self.width, height=self.height, batch_size=self.batch_size
        )
        self.tensors[device_id] = tensors
        return tensors.create_io_tensors(
            input_types=dtype,
            input_devices=device_id,
            output_types=dtype,
            output_devices=device_id,
        )

    def compute_timesteps(self):
        if self.is_fraction:
            if not self.timesteps:
                for current_index in range(self.total_frames + 1):
                    factor = Fraction(self.vfi_factor).limit_denominator(100)
                    vfi_factor_numerator = factor.numerator
                    vfi_factor_denominator = factor.denominator
                    next_index = current_index + 1
                    output_start = (
                        current_index * vfi_factor_numerator
                    ) // vfi_factor_denominator
                    output_end = (
                        next_index * vfi_factor_numerator
                    ) // vfi_factor_denominator
                    self.timesteps.append(
                        [
                            (
                                output_start
                                + i * vfi_factor_denominator % vfi_factor_numerator
                            )
                            / vfi_factor_numerator
                            for i in range(1, output_end - output_start)
                        ]
                    )
            return self.timesteps
        else:
            if not self.timesteps:
                self.timesteps = [
                    [i / self.vfi_factor for i in range(1, self.vfi_factor)]  # type: ignore
                ]
            return self.timesteps[0]

    # @override
    # def inference(self, frame: Tensor, **kwargs) -> list[Tensor]:
    #     """Inference with batching"""
    #     self.frame_counter += 1
    #     len_img0 = len(self.img0_frame_buffer)
    #     len_img1 = len(self.img1_frame_buffer)
    #     if len_img0 < self.batch_size or len_img0 != len_img1:
    #         if len_img0 == 0:
    #             self.img0_frame_buffer.append(frame)
    #         elif len_img0 == len_img1:
    #             self.img0_frame_buffer.extend([self.img1_frame_buffer[-1], frame])
    #             self.img1_frame_buffer.append(frame)
    #         else:
    #             self.img1_frame_buffer.append(frame)
    #         return []
    #     else:
    #         interpolated = self.vfi_model.do_vfi(
    #             torch.stack(self.img0_frame_buffer),
    #             torch.stack(self.img1_frame_buffer),
    #             [0.5],
    #             # self.get_timesteps()[self.frame_counter]
    #             # if self.is_fraction
    #             # else self.get_timesteps(),  # type: ignore
    #         )

    #         out = []
    #         for i, original_frame in enumerate(self.img0_frame_buffer):
    #             out.extend([original_frame, *interpolated[i]])

    #         self.img0_frame_buffer.clear()
    #         self.img0_frame_buffer.append(self.img1_frame_buffer[-1])
    #         self.img1_frame_buffer.clear()
    #         self.img1_frame_buffer.append(frame)

    #         return out

    @override
    def inference(self, frame: Tensor, **kwargs) -> list[Tensor]:
        """
        Inference with frame batching.
        
        NOTE: This method uses 5D timestep tensors and expects the model to accept those
        and return predictions as 5D frame tensors.
        The tensor shapes are as follows:
        ```
        timestep = (timestep_amount, batch_size, 1, height, width)
        prediction =  (timestep_amount, batch_size, 3, height, width)
        ```
        
        Example
        --------
        A frame batch looks like:
        ```
        batch_size = 5
        # 'f' means a frame tensor
        img0 = [f0, f1, f2, f3, f4]
        img1 = [f1, f2, f3, f4, f5]
        ```
        """
        self.frame_counter += 1
        len_img0 = len(self.img0_frame_buffer)
        len_img1 = len(self.img1_frame_buffer)
        if len_img0 < self.batch_size or len_img0 != len_img1:
            if len_img0 == 0:
                self.img0_frame_buffer.append(frame)
            elif len_img0 == len_img1:
                self.img0_frame_buffer.extend([self.img1_frame_buffer[-1], frame])
                self.img1_frame_buffer.append(frame)
            else:
                self.img1_frame_buffer.append(frame)
            return []
        else:
            # TODO: Remember to assign the cuda stream in the cuda options etc.
            current_device_id = self.devices[
                self.frame_batch_counter % self.device_count
            ]
            current_tensors = self.tensors[current_device_id]
            io_binding = self.io_bindings[current_device_id]
            timesteps = compute_timesteps(self.vfi_factor, self.frame_counter)
            
            current_tensors.I0_IN.fill_(torch.stack(self.img0_frame_buffer))
            current_tensors.I1_IN.fill_(torch.stack(self.img1_frame_buffer))
            current_tensors.TIMESTEP_IN.fill_(torch.stack([current_tensors.TIMESTEP_IN.]))
            
            for timestep in compute_timesteps(self.vfi_factor, self.frame_counter):
                current_tensors.TIMESTEP_IN.fill_(timestep)
                
                io_binding.synchronize_inputs()
                self.sessions[current_device_id].run_with_iobinding(io_binding)
                io_binding.synchronize_outputs()

            interpolated = self.vfi_model.do_vfi(
                ,
                torch.stack(self.img1_frame_buffer),
                [0.5],
                # self.get_timesteps()[self.frame_counter]
                # if self.is_fraction
                # else self.get_timesteps(),  # type: ignore
            )

            out = []
            for i, original_frame in enumerate(self.img0_frame_buffer):
                out.extend([original_frame, *interpolated[i]])

            self.img0_frame_buffer.clear()
            self.img0_frame_buffer.append(self.img1_frame_buffer[-1])
            self.img1_frame_buffer.clear()
            self.img1_frame_buffer.append(frame)

            return out
