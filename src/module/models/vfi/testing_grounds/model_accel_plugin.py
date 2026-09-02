import logging
import os
import queue
import sys
import time
from pathlib import Path
from queue import Queue
from threading import Lock

import gi
import numpy as np
import torch
import torchvision
from torch import Tensor

logger = logging.getLogger(__file__)
logger.propagate = False
logger.setLevel("DEBUG")
console_formatter = logging.Formatter(
    "%(asctime)s - %(module)s - %(lineno)s - %(levelname)s - %(message)s"
)

console_handler = logging.StreamHandler()
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)


# Set application path
os.environ["TAS_PATH"] = f"{os.path.dirname(os.path.abspath(__file__))}"

from src.module.models.vfi.vfi_base import VFIModelBase

gi.require_version("Gst", "1.0")

from gi.repository import GLib, GObject, Gst  # type: ignore

# WARNING: This function will terminate your program if it was unable to initialize GStreamer for some reason.
# If you want your program to fall back, use Gst.init_check instead.
Gst.init()


frame_format, pixel_bytes = "RGBA", 4
device = "cuda"
dtype = torch.float32
preprocess = torchvision.transforms.ToTensor()

model = VFIModelBase(
    model_path=Path(
        "/home/cachy/Programming_Projects/TheAnimeScripter/weights/vfi/rife_elexor.pth"
    ),
    width=1920,
    height=1080,
    vfi_factor=2,
    scale=1,
    batch_size=1,
    total_frames=0,
    device=device,
    dtype=dtype,
)


# num-buffers=200 defines how many frames will be published by a given element. After sending "num-buffers", EOS (end-of-stream) event is published. num-buffers=-1 means all frames
# The "-e" flag ensures the output file is written correctly
# The queue blocks caller until more space is available. The default queue size limits are 200 buffers, 10MB of data, or one second worth of data, whichever is reached first.
# pipeline = Gst.parse_launch(f"""
#     filesrc location=media/in.mkv num-buffers=200 !
#     decodebin !
#     videoconvert !
#     video/x-raw,format={frame_format} !
#     vfifilter !
#     fakesink name=s
# """)
#     queue !
# nvh265enc preset=p1 const-quality=13 !
# matroskamux !
# filesink name=s location=media/out.mkv
# video/x-raw,format=I420_10LE,framerate=48/1.001


# model = VFIModel(width=1920, height=1080, vfi_factor=2, scale=1)

some_time = 60 * (10**9)  # 60 seconds
fps_n = 48000
fps_d = 1001
# Splitting input into multiple files: https://gstreamer.freedesktop.org/documentation/multifile/splitmuxsink.html?gi-language=python
pipeline = Gst.parse_launch(f"""
    filesrc location=media/Bofuri_S2E4.mkv num-buffers=200 !
    decodebin !
    videoconvert !
    videorate name=s !
    video/x-raw,format={frame_format},framerate={fps_n}/{fps_d} !
    videoconvert name=o !
    x265enc key-int-max=10 tune=animation !
    h265parse !
    matroskamux name=fullVideoMux ! 
    filesink location=media/fullvideo.mkv
""")
# option-string="preset=veryfast:crf=15:profile=main10"
# splitmuxsink location=media/inf_out%06d.mkv async-finalize=true max-size-time={time} muxer-factory=matroskamux"
# fakesink name=s
# nvh265enc preset=p1 const-quality=13 !
#  h265parse !
# ,framerate=48000/1001
# Commands
#   gst-launch-1.0 --gst-plugin-path="/home/cachy/Programming_Projects/TheAnimeScripter/src/module/models/vfi/testing_grounds" filesrc location=media/Bofuri_S2E4.mkv num-buffers=100 ! decodebin ! videoconvert ! video/x-raw,format={frame_format} ! vfifilter ! nvh265enc preset=p1 const-quality=13 ! matroskamux ! filesink name=s location=media/inf_out.mkv
#   PYTHONPATH="/home/cachy/Programming_Projects/TheAnimeScripter/.venv/lib/python3.13/site-packages" gst-launch-1.0 --gst-plugin-path="/home/cachy/Programming_Projects/TheAnimeScripter/src/module/models/vfi/testing_grounds" filesrc location="/home/cachy/Programming_Projects/TheAnimeScripter/media/Bofuri_S2E4.mkv" num-buffers=100 ! decodebin ! videoconvert ! video/x-raw,format={frame_format} ! vfifilter ! nvh265enc preset=p1 const-quality=13 ! matroskamux ! filesink name=s location="/home/cachy/Programming_Projects/TheAnimeScripter/media/inf_out.mkv"

output_pad: Gst.Pad = pipeline.get_by_name("s").get_static_pad("src")
input_pad: Gst.Pad = pipeline.get_by_name("o").get_static_pad("src")

# new_caps_filter = Gst.caps_from_string(
#     "video/x-raw, format=RGB, width=1920, height=1080, framerate=48000/1001"
# )
# caps_query = Gst.Query.new_caps(new_caps_filter)
# success = output_pad.query(caps_query)
# if not success:
#     print("failed to set caps")

frame_counter: int = 0
in_frame_buffer: Queue[Tensor] = Queue(maxsize=16)
out_buffers: Queue[Gst.Buffer] = Queue()
out_frame_buffer: Queue[tuple[Tensor, Gst.Buffer]] = Queue()
lock = Lock()

duration = 1 / (fps_n / fps_d)
next_time = duration


def empty_buffer(reference_buffer: Gst.Buffer) -> Gst.Buffer:
    buffer = reference_buffer
    out_buffer = Gst.Buffer.new()

    out_buffer.pts = buffer.pts
    out_buffer.dts = buffer.dts
    out_buffer.duration = buffer.duration
    out_buffer.offset = buffer.offset
    out_buffer.offset_end = buffer.offset_end
    return out_buffer


def push_pad(pad: Gst.Pad, buffer: Gst.Buffer) -> None:
    logger.debug("Pushing pad")
    ret = pad.push(buffer)
    match ret:
        case Gst.FlowReturn.ERROR:
            logger.error("failed to push buffer")
        case Gst.FlowReturn.NOT_NEGOTIATED:
            logger.error("NOT_NEGOTIATED")
        case Gst.FlowReturn.NOT_SUPPORTED:
            logger.error("NOT_SUPPORTED")
        case Gst.FlowReturn.FLUSHING:
            logger.error("FLUSHING")


def tensor_to_buffer(tensor: Tensor, reference_buffer: Gst.Buffer) -> Gst.Buffer:
    global frame_counter, next_time
    buffer = reference_buffer
    out_buffer = Gst.Buffer.new_wrapped(tensor.numpy().tobytes())

    out_buffer.pts = buffer.pts
    out_buffer.dts = buffer.dts
    out_buffer.duration = buffer.duration
    out_buffer.offset = buffer.offset
    out_buffer.offset_end = buffer.offset_end
    # print(next_time)
    # out_buffer.pts = next_time
    # out_buffer.duration = duration
    # out_buffer.offset = frame_counter
    # frame_counter += 1
    # out_buffer.offset_end = frame_counter

    # next_time += duration
    return out_buffer


def inplace_tensor_to_buffer(tensor: Tensor, buffer: Gst.Buffer) -> None:
    buffer.fill(0, tensor.numpy().tobytes())


def on_input_pad(pad: Gst.Pad, info: Gst.PadProbeInfo):
    buffer = info.get_buffer()
    image_tensor = buffer_to_image_tensor(buffer, pad.get_current_caps())
    in_frame_buffer.put(image_tensor)
    return Gst.PadProbeReturn.OK


def on_output_pad(pad: Gst.Pad, info: Gst.PadProbeInfo):
    """_summary_

    Parameters
    ----------
    pad : Gst.Pad
        The pad that called this function.
        Think of it as `self` (like on classes).
    info : Gst.PadProbeInfo
        The data we're probing.

    Returns
    -------
    Gst.PadProbeReturn
        Different return values for the Gst.PadProbeCallback.
    """
    logger.debug("start")
    buffer = info.get_buffer()
    logger.debug(f"[{buffer.offset}]: ({buffer.pts / Gst.SECOND:6.2f})")

    if out_frame_buffer.qsize() > 0:
        while True:
            try:
                out_frame, _buffer = out_frame_buffer.get_nowait()
            except queue.Empty:
                break
            out_buffer = tensor_to_buffer(out_frame, _buffer)
            peer_pad = pad.get_peer()
            push_pad(peer_pad, out_buffer)

            # inplace_tensor_to_buffer(out_frame, _buffer)
            # push_pad(pad, _buffer)
    try:
        in_frame = in_frame_buffer.get_nowait()
        outputs = model.inference(in_frame)
        out_buffers.put(buffer)
        logger.debug(f"VFI: {len(outputs)}")
        for vfi_frame in outputs:
            out_frame_buffer.put((vfi_frame.cpu(), out_buffers.get_nowait()))
        torch.cuda.synchronize()
    except queue.Empty:
        out_buffers.put(buffer)  # Store buffer for later
        logger.debug("end")
        return Gst.PadProbeReturn.HANDLED
    logger.debug("end")

    return Gst.PadProbeReturn.HANDLED


def on_frame_probe(pad: Gst.Pad, info: Gst.PadProbeInfo):
    """_summary_

    Parameters
    ----------
    pad : Gst.Pad
        The pad that called this function.
        Think of it as `self` (like on classes).
    info : Gst.PadProbeInfo
        The data we're probing.

    Returns
    -------
    Gst.PadProbeReturn
        Different return values for the Gst.PadProbeCallback.
    """
    buffer = info.get_buffer()
    # print(f"[{buffer.pts / Gst.SECOND:6.2f}]")

    with lock:
        image_tensor = buffer_to_image_tensor(buffer, pad.get_current_caps())
        # try:
        #     image_tensor = frame_buffer.get_nowait()
        # except queue.Empty:
        #     print("NO FRAME")
        #     return Gst.PadProbeReturn.OK

        outputs = model.inference(image_tensor)
    rets = []

    rets.append(output_pad.push(buffer))
    for vfi_frame in outputs:
        vfi_frame_bytes = vfi_frame.cpu().numpy().tobytes()
        vfi_buffer = Gst.Buffer.new_wrapped(vfi_frame_bytes)
        rets.append(pad.push(vfi_buffer))

    for ret in rets:
        match ret:
            case Gst.FlowReturn.ERROR:
                print("failed to push buffer")
            case Gst.FlowReturn.NOT_NEGOTIATED:
                print("NOT_NEGOTIATED")
            case Gst.FlowReturn.NOT_SUPPORTED:
                print("NOT_SUPPORTED")

    return Gst.PadProbeReturn.HANDLED


def buffer_to_image_tensor(buffer: Gst.Buffer, caps: Gst.Caps) -> Tensor:
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
            return preprocess(image_array[:, :, :3]).to(
                dtype=dtype, device=device
            )  # RGBA -> RGB
        finally:
            buffer.unmap(map_info)


pipeline.get_by_name("s").get_static_pad("src").add_probe(  # Output
    Gst.PadProbeType.BUFFER, on_output_pad
)
pipeline.get_by_name("s").get_static_pad("sink").add_probe(  # Input
    Gst.PadProbeType.BUFFER, on_input_pad
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
        else:
            print("No msg")
finally:
    # with open(f"logs/{os.path.splitext(sys.argv[0])[0]}.pipeline.dot", "x") as file:
    #     file.write(Gst.debug_bin_to_dot_data(pipeline, Gst.DebugGraphDetails.ALL))
    pipeline.set_state(Gst.State.NULL)
