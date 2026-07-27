import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

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

    def forward(self, x: Tensor, flow: Tensor | None = None, scale=1):
        x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear")
        if flow is not None:
            flow = (
                F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear")
                * 1.0
                / scale
            )
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear")
        flow = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        return flow, mask


# multiplier=32
class IFNet(nn.Module):
    def __init__(
        self,
        width: int,
        height: int,
        padded_width: int,
        padded_height: int,
        scale: int | float = 1,
        dtype: torch.dtype = torch.float32,
        device: torch.types.Device = "cuda",
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.padded_width = padded_width
        self.padded_height = padded_height
        self.ensemble = ensemble
        self.block0 = IFBlock(7, c=192)
        self.block1 = IFBlock(8 + 4, c=128)
        self.block2 = IFBlock(8 + 4, c=96)
        self.block3 = IFBlock(8 + 4, c=64)
        self.scaleList = [8 / scale, 4 / scale, 2 / scale, 1 / scale]
        self.blocks = [self.block0, self.block1, self.block2, self.block3]
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

    def forward(self, img0: Tensor, img1: Tensor, timestep: Tensor):
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        mask = None

        for i, block in enumerate(self.blocks):
            scale = self.scaleList[i]

            if flow is None:
                flow, mask = block(
                    torch.cat((img0[:, :3], img1[:, :3], timestep), 1),
                    None,
                    scale=scale,
                )

                if self.ensemble:
                    f1, m1 = block(
                        torch.cat((img1[:, :3], img0[:, :3], 1 - timestep), 1),
                        None,
                        scale=scale,
                    )
                    flow = (flow + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
                    mask = (mask - m1) / 2
            else:
                f0, m0 = block(
                    torch.cat(
                        (warped_img0[:, :3], warped_img1[:, :3], timestep, mask), 1
                    ),
                    flow,
                    scale=scale,
                )

                if self.ensemble:
                    f1, m1 = block(
                        torch.cat(
                            (
                                warped_img1[:, :3],
                                warped_img0[:, :3],
                                1 - timestep,
                                -mask,
                            ),  # type: ignore
                            1,
                        ),
                        torch.cat((flow[:, 2:4], flow[:, :2]), 1),
                        scale=scale,
                    )
                    f0 = (f0 + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
                    m0 = (m0 - m1) / 2

                flow = flow + f0
                mask = mask + m0

            warped_img0 = warp(img0, flow[:, :2], self.ten_flow, self.back_warp)
            warped_img1 = warp(img1, flow[:, 2:4], self.ten_flow, self.back_warp)

        temp = torch.sigmoid(mask)  # type: ignore
        return (warped_img0 * temp + warped_img1 * (1 - temp))[
            :, :, : self.height, : self.width
        ]
