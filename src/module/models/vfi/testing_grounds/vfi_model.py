import torch
from torch import Tensor
from torch.types import Device


class VFIModel:
    def __init__(
        self,
        batch_size: int,
        width: int,
        height: int,
        vfi_factor: int | float,
        scale: float,
        dtype: torch.dtype,
        device: Device,
    ) -> None:
        self.frame_width = width
        self.frame_height = height
        self.vfi_factor = vfi_factor
        self.scale = scale
        self.model_dtype = dtype
        self.device = device

        from src.module.models.vfi.rife.IFNet_elexor_basic import IFNet

        self.vfi_model = IFNet(
            width=self.frame_width,
            height=self.frame_height,
            vfi_factor=self.vfi_factor,
            scale=self.scale,
            dtype=self.model_dtype,
            device=self.device,
        )
        self.vfi_model.load_state_dict(
            torch.load(
                "/home/cachy/Programming_Projects/TheAnimeScripter/weights/vfi/rife_elexor.pth"
            )
        )
        self.vfi_model.eval().to(self.device, dtype=self.model_dtype)

        self.img0: Tensor | None = None
        self.img1: Tensor | None = None
        self.is_fraction = isinstance(self.vfi_factor, float)
        self.TIMESTEP = torch.full(
            [
                batch_size,
                1,
                self.frame_height,
                self.frame_width,
            ],
            vfi_factor,
            dtype=self.model_dtype,
            device=self.device,
        )

        self.stream = torch.cuda.Stream()

    def do_vfi(
        self, img0: Tensor, img1: Tensor, timesteps: list[float]
    ) -> list[list[Tensor]]:
        out: list[list[Tensor]] = []
        with torch.inference_mode():
            with torch.cuda.stream(self.stream):
                for timestep in timesteps:
                    interpolated = self.vfi_model(
                        img0,
                        img1,
                        self.TIMESTEP.fill_(  # FIXME: Timesteps should be filled with different values in a batch, e.g. [[0.4, 0.6], [0.5]], to match the frames in a batch. Do not fill with a single value as done here
                            timestep
                        )  # Timesteps only change between frames when vfi factor is a fraction.
                        if self.is_fraction
                        else self.TIMESTEP,
                    )  # type: Tensor
                    self.stream.synchronize()

                    for i, vfi_frame in enumerate(interpolated.unbind()):
                        try:
                            out[i].append(vfi_frame)
                        except IndexError:
                            out.append([vfi_frame])
        return out
