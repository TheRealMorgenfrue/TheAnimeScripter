import contextlib
import math
import os
import sys
import time

import cv2
import gi
from torch.multiprocessing import Queue

gi.require_version("Gst", "1.0")
import numpy as np
import torch
import torchvision
from gi.repository import Gst  # type: ignore
from torch import Tensor

frame_width, frame_height = 1920, 1080
frame_format, pixel_bytes, model_precision = "RGBA", 4, "fp32"
model_dtype = torch.float16 if model_precision == "fp16" else torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from src.module.models.vfi.rife.IFNet_elexor_basic import IFNet

vfi_model = IFNet(
    width=frame_width,
    height=frame_height,
    vfi_factor=2,
    scale=1,
    dtype=model_dtype,
    device=device,
)
vfi_model.load_state_dict(
    torch.load(
        "/home/cachy/Programming_Projects/TheAnimeScripter/weights/vfi/rife_elexor.pth"
    )
)
vfi_model.eval().to(device)
preprocess = torchvision.transforms.ToTensor()

img0: Tensor = None
img1: Tensor = None
TIMESTEP = torch.full(
    [1, 1, frame_height, frame_width], 0.5, dtype=model_dtype, device=device
)

OUT_BUFFER = Queue()
frame_format = "RGBA"


Gst.init()
pipeline = Gst.parse_launch(f"""
    filesrc location=media/in.mkv num-buffers=200 !
    decodebin !
    videoconvert !
    video/x-raw,format={frame_format} !
    fakesink name=s
""")


def on_frame_probe(pad, info):
    global img0, img1, OUT_BUFFER

    buffer = info.get_buffer()
    print(f"[{buffer.pts / Gst.SECOND:6.2f}]")

    image_tensor = buffer_to_image_tensor(buffer, pad.get_current_caps())
    image_batch = image_tensor.unsqueeze(0).to(device)

    if img0 is None:
        img0 = image_batch
    elif img1 is None:
        img1 = image_batch

        with torch.no_grad():
            interpolated = vfi_model(img0, img1, TIMESTEP)
            OUT_BUFFER.put_nowait(interpolated.cpu().numpy())
            img0 = img1
            img1 = None
    return Gst.PadProbeReturn.OK


def buffer_to_image_tensor(buffer, caps) -> Tensor:
    caps_structure = caps.get_structure(0)
    height, width = (
        caps_structure.get_value("height"),
        caps_structure.get_value("width"),
    )

    is_mapped, map_info = buffer.map(Gst.MapFlags.READ)
    if is_mapped:
        try:
            image_array = np.ndarray(
                (height, width, pixel_bytes), dtype=np.uint8, buffer=map_info.data
            ).copy()  # extend array lifetime beyond subsequent unmap
            return preprocess(image_array[:, :, :3])  # RGBA -> RGB
        finally:
            buffer.unmap(map_info)


def write_file():
    fps = 48
    gst_str = "'appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=3000 speed-preset=superfast ! filesink location=media/out.mkv"
    out = cv2.VideoWriter(
        gst_str, cv2.CAP_GSTREAMER, 0, fps, (frame_width, frame_height), True
    )
    if not out.isOpened():
        print("failed to open video writer")
    print("writing file")
    while True:
        try:
            img = OUT_BUFFER.get_nowait()
            out.write(img)
        except Exception:
            break
    out.release()


pipeline.get_by_name("s").get_static_pad("sink").add_probe(
    Gst.PadProbeType.BUFFER, on_frame_probe
)

pipeline.set_state(Gst.State.PLAYING)

try:
    while True:
        msg = pipeline.get_bus().timed_pop_filtered(
            Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR
        )
        if msg:
            text = msg.get_structure().to_string() if msg.get_structure() else ""
            msg_type = Gst.message_type_get_name(msg.type)
            print(f"{msg.src.name}: [{msg_type}] {text}")
            break
    write_file()
finally:
    # with open(f"logs/{os.path.splitext(sys.argv[0])[0]}.pipeline.dot", "x") as file:
    #     file.write(Gst.debug_bin_to_dot_data(pipeline, Gst.DebugGraphDetails.ALL))
    pipeline.set_state(Gst.State.NULL)
