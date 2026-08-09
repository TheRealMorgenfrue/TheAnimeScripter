import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from module.models.vfi.vfi_utils import compute_resolution_padding

from .util.warplayer_v2 import warp


def conv(
    in_planes: int, out_planes: int, kernel_size=3, stride=1, padding=1, dilation=1
):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode="replicate",
            dilation=dilation,
            bias=True,
        ),
        nn.LeakyReLU(0.2, True),
    )


class ResConv(nn.Module):
    def __init__(self, c: int, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(
            torch.ones((1, c, 1, 1)), requires_grad=True
        )  # REVIEW: Should this be grad?
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock(nn.Module):
    def __init__(self, in_planes: int, c=64):
        super().__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
        )
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(c, 4 * 6, 4, 2, 1), nn.PixelShuffle(2)
        )

    def forward(self, x: Tensor, h: int, w: int, flow: Tensor | None = None, scale=1):
        x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear")

        if flow is not None:
            flow = (
                F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear") / scale
            )
            x = torch.cat((x, flow), 1)

        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        tmp = F.interpolate(tmp, size=(h, w), mode="bilinear")

        flow = tmp[:, :4] * scale
        mask = tmp[:, 4:5]

        return flow, mask


# multiplier=64
class IFNet(nn.Module):
    def __init__(
        self,
        width: int,
        height: int,
        vfi_factor: int | float,
        scale: int | float,
        dtype: torch.dtype,
        device: torch.types.Device,
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.padded_width = compute_resolution_padding(width, 64)
        self.padded_height = compute_resolution_padding(height, 64)
        self.vfi_factor = vfi_factor
        self.block0 = IFBlock(7 + 8, c=192)
        self.block1 = IFBlock(8 + 4 + 8, c=128)
        self.block2 = IFBlock(8 + 4 + 8, c=96)
        self.block3 = IFBlock(8 + 4 + 8, c=64)
        self.encode = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.ConvTranspose2d(16, 4, 4, 2, 1)
        )
        self.scale_list = [
            32 / scale,
            16 / scale,
            8 / scale,
            4 / scale,
            2 / scale,
            1 / scale,
        ]
        self.blocks = [
            self.block0,
            self.block1,
            self.block0,
            self.block1,
            self.block2,
            self.block3,
        ]
        self.ten_flow = torch.tensor(
            [(self.padded_width - 1.0) / 2.0, (self.padded_height - 1.0) / 2.0],
            dtype=dtype,
            device=device,
        )
        ten_horizontal = (
            torch.linspace(-1.0, 1.0, self.padded_width, dtype=dtype, device=device)
            .view(1, 1, 1, self.padded_width)
            .expand(-1, -1, self.padded_height, -1)
        ).to(dtype=dtype, device=device)
        ten_vertical = (
            torch.linspace(-1.0, 1.0, self.padded_height, dtype=dtype, device=device)
            .view(1, 1, self.padded_height, 1)
            .expand(-1, -1, -1, self.padded_width)
        ).to(dtype=dtype, device=device)
        self.back_warp = torch.cat([ten_horizontal, ten_vertical], 1)
        self.f0 = None
        self.f1 = None
        self.counter = 1

    def cache(self):
        assert self.f0 is not None
        self.f0.copy_(self.f1, non_blocking=True)

    def cacheReset(self, frame):
        self.f0 = self.encode(frame)

    def forward(self, img0, img1, timestep):
        if self.counter == self.vfi_factor:
            self.counter = 1
            if self.f0 is None:
                self.f0 = self.encode(img0)
            self.f1 = self.encode(img1)
        else:
            if self.f0 is None or self.f1 is None:
                self.f0 = self.encode(img0)
                self.f1 = self.encode(img1)
        self.counter += 1

        warped_img0 = img0
        warped_img1 = img1
        flow = None
        large_flow = None
        mask = None

        for i in range(6):
            if flow is None:
                flow, mask = self.blocks[i](
                    torch.cat((img0, img1, self.f0, self.f1, timestep), 1),
                    self.padded_height,
                    self.padded_width,
                    None,
                    scale=self.scale_list[i],
                )

                if large_flow is not None:
                    magnitude = torch.sqrt(
                        large_flow[:, 0, :, :] ** 2 + large_flow[:, 1, :, :] ** 2
                    )
                    count = torch.sum(magnitude > 40)
                    mask_large = count > 1036800
                    flow = torch.where(mask_large, large_flow, flow)
            else:
                wf0 = warp(self.f0, flow[:, :2], self.ten_flow, self.back_warp)
                wf1 = warp(self.f1, flow[:, 2:4], self.ten_flow, self.back_warp)
                fd, mask = self.blocks[i](
                    torch.cat((warped_img0, warped_img1, wf0, wf1, timestep, mask), 1),
                    self.padded_height,
                    self.padded_width,
                    flow,
                    scale=self.scale_list[i],
                )
                flow = flow + fd

            warped_img0 = warp(img0, flow[:, :2], self.ten_flow, self.back_warp)
            warped_img1 = warp(img1, flow[:, 2:4], self.ten_flow, self.back_warp)

            if i == 1:
                large_flow = flow
                flow = None

        mask = torch.sigmoid(mask)  # type: ignore
        return (warped_img0 * mask + warped_img1 * (1 - mask))[
            :, :, : self.height, : self.width
        ]
