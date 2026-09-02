from collections import deque
from fractions import Fraction
from typing import Any

print("importing gi in model")
import gi

print("done")
import numpy as np
import torch
import torchvision
from numpy.typing import NDArray

from src.module.models.vfi.testing_grounds.vfi_model import VFIModel

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
gi.require_version("GstVideo", "1.0")

from gi.repository import GLib, GObject, Gst, GstBase, GstVideo  # type: ignore

IN_CAPS = Gst.Caps.from_string(
    "video/x-raw, format=RGBA, "
    f"width=(int)[ 1, {GLib.MAXINT} ], height=(int)[ 1, {GLib.MAXINT} ], "
    f"framerate=(fraction)[ 1/1, {GLib.MAXINT}/1 ]"
)
"""Plugin output"""

OUT_CAPS = Gst.Caps.from_string(
    "video/x-raw, format=RGB, "
    f"width=(int)[ 1, {GLib.MAXINT} ], height=(int)[ 1, {GLib.MAXINT} ], "
    f"framerate=(fraction)[ 1/1, {GLib.MAXINT}/1 ]"
)
"""Plugin input"""

SRC_PAD_TEMPLATE = Gst.PadTemplate.new(  # Plugin output
    "src", Gst.PadDirection.SRC, Gst.PadPresence.ALWAYS, OUT_CAPS
)
"""Plugin output"""

SINK_PAD_TEMPLATE = Gst.PadTemplate.new(  # Plugin input
    "sink", Gst.PadDirection.SINK, Gst.PadPresence.ALWAYS, IN_CAPS
)
"""Plugin input"""


class VFIFilter(GstBase.BaseTransform):
    __gstmetadata__ = (
        "vfifilter",  # name
        "Filter",  # classification
        "Apply VFI models to the input",  # description
        "Morgenfrue",  # author
    )

    __gsttemplates__ = (SRC_PAD_TEMPLATE, SINK_PAD_TEMPLATE)

    __gproperties__ = {
        "vfi_factor": (
            GObject.TYPE_FLOAT,  # GObject.TYPE_*
            "Interpolation factor",  # str
            "Increases framerate by this factor by inserting intermediate frames",  # str
            1.0,  # min value
            GLib.MAXFLOAT,  # max value
            2.0,  # default value
            GObject.ParamFlags.READWRITE,  # GObject.ParamFlags
        ),
        "scale": (
            GObject.TYPE_FLOAT,  # GObject.TYPE_*
            "Interpolation frame scale",  # str
            "Scales frames by this factor before interpolation",  # str
            0.25,  # min value
            2.0,  # max value
            1.0,  # default value
            GObject.ParamFlags.READWRITE,  # GObject.ParamFlags
        ),
    }

    def __init__(self) -> None:
        super().__init__()
        self.set_qos_enabled(True)
        self.set_passthrough(False)
        self.passthrough = False
        self._initialized = False
        self.frame_buffer: deque[NDArray]
        self.should_get_next_frame: bool
        self.width: int
        self.height: int
        self.fps_numerator: int
        self.fps_denominator: int
        self.vfi_factor: float
        self.scale: float
        self.frame_duration: int
        self.next_time: int

    def do_start(self) -> None:
        """Called when the pipeline is starting. Elements should make expensive calls here."""
        print("do_start")
        if not self.is_passthrough:
            if not self._initialized:
                print("init VFI model")
                self._vfi_model = VFIModel(
                    batch_size=1,
                    width=self.width,
                    height=self.height,
                    vfi_factor=self.vfi_factor,
                    scale=self.scale,
                    dtype=torch.float32,
                    device="cuda",
                )
                self.preprocess = torchvision.transforms.ToTensor()
                self.pixel_bytes = 4  # RGBA=4, RGB=3
                self._initialized = True
        else:
            print("VFI passthrough mode engaged")

    def do_stop(self) -> None:
        if hasattr(self, "_vfi_model"):
            del self._vfi_model

    def do_get_property(self, prop: GObject.GParamSpec) -> Any:
        if prop.name == "vfi_factor":
            return self.vfi_factor
        elif prop.name == "scale":
            return self.scale
        else:
            raise AttributeError(f"unknown property '{prop.name}'")

    def do_set_property(self, prop: GObject.GParamSpec, value: Any) -> None:
        if prop.name == "vfi_factor":
            self.vfi_factor = value
        elif prop.name == "scale":
            self.scale = value
        else:
            raise AttributeError(f"unknown property '{prop.name}'")

    def do_set_caps(self, incaps: Gst.Caps, outcaps: Gst.Caps) -> bool:
        """
        Note: Use do_set_caps to set plugin in passthrough mode.

        This function is only called when caps are finalized.

        See caps negotiation: https://gstreamer.freedesktop.org/documentation/additional/design/element-transform.html#negotiation
        """
        print("do_set_caps")
        in_info = GstVideo.VideoInfo()
        in_info.from_caps(incaps)
        out_info = GstVideo.VideoInfo()
        out_info.from_caps(outcaps)

        # if input_framerate == output_framerate set plugin to passthrough mode
        print(
            f"setting caps: {in_info.fps_n / in_info.fps_d * self.vfi_factor}=={out_info.fps_n / out_info.fps_d * self.vfi_factor}"
        )
        if (
            in_info.fps_n / in_info.fps_d * self.vfi_factor
            == out_info.fps_n / out_info.fps_d * self.vfi_factor
        ):
            # If there is no transform_ip function in passthrough mode, the buffer is pushed intact
            self.set_passthrough(True)
        else:
            self.frame_buffer = deque()
            self.width = in_info.width
            self.height = in_info.height
            self.fps_numerator = in_info.fps_n
            self.fps_denominator = in_info.fps_d

            self.frame_duration = Gst.util_uint64_scale_int(
                Gst.SECOND, out_info.fps_d, out_info.fps_n
            )
            self.next_time = self.frame_duration

        # in_struct = incaps.get_structure(0)
        # out_struct = outcaps.get_structure(0)
        # self.width, self.height = [in_struct.get_value(v) for v in ["width", "height"]]
        # (_, self.frame_numerator, self.frame_denominator) = in_struct.get_fraction(
        #     "framerate"
        # )
        # (_, out_frame_numerator, out_frame_denominator) = out_struct.get_fraction(
        #     "framerate"
        # )
        # # if input framerate == output framerate set plugin to passthrough mode
        # if (
        #     self.frame_numerator / self.frame_denominator * self.vfi_factor
        #     == out_frame_numerator / out_frame_denominator * self.vfi_factor
        # ):
        #     self.set_passthrough(True)

        return True

    # REVIEW: Maybe it's not necessary to make a plugin (in that case disregard todos below)
    # If a pipeline is manually constructed with src/sink pads, it may be possible to send a query with new caps on an appropriate pad.
    # Then install a probe on the same pad and push additional buffers to the next pad in the pipeline (if buffers are 1-N).
    # If buffers are 1-1, then just transform the input in-place.
    # Query: https://gstreamer.freedesktop.org/documentation/gstreamer/gstquery.html?gi-language=python#GstQuery
    # Pad query: https://gstreamer.freedesktop.org/documentation/gstreamer/gstpad.html?gi-language=python#gst_pad_query
    #
    # Another approach could also be to define a chain function and do processing there. Investigate this further.
    # Chain: https://gstreamer.freedesktop.org/documentation/plugin-development/basics/chainfn.html?gi-language=python
    #
    # A third approach could be to implement dual gstreamer pipelines: https://gstreamer.freedesktop.org/documentation/gstreamer/gstpad.html#GstPadProbeReturn
    #
    # TODO: Use Fixed or Transform caps: https://gstreamer.freedesktop.org/documentation/plugin-development/advanced/negotiation.html?gi-language=python
    # TODO: And implment event handling. And set caps directly on source pad
    # TODO: Then it might just be possible to push extra frames directly to the source pad
    # TODO: https://gstreamer.freedesktop.org/documentation/plugin-development/advanced/negotiation.html#implementing-a-caps-query-function
    # TODO: Rust is not currently ready for ML; use Python with Gstreamer. However, Pascal arch and Python14 are not compatible.
    # Compile onnxruntime-gpu==1.20.2 with Python14 and provide a wheel on GitHub
    def do_transform_caps(
        self, direction: Gst.PadDirection, caps: Gst.Caps, filter_: Gst.Caps
    ) -> Gst.Caps:
        """
        The base class, BaseTransform, assumes that input and output caps will depend on each other.

        However, if this is not the case this function, do_transform_caps,
        """
        print("do_transform_caps caps")
        if direction == Gst.PadDirection.SRC:
            res = IN_CAPS
        else:
            res = OUT_CAPS

        # intersect caps if there is transform
        if filter_:
            # create new caps that contains all formats that are common to both
            res = res.intersect(filter_)
        print(res.serialize(Gst.SerializeFlags.NONE))
        return res

    def do_fixate_caps(
        self, direction: Gst.PadDirection, caps: Gst.Caps, othercaps: Gst.Caps
    ) -> Gst.Caps:
        """
        caps: initial caps
        othercaps: target caps

        NOTE: If for example downstream (source pad) also accepts a wide range of resolutions,
        the default behaviour of the base class will be to pick the smallest possible resolution.
        Thus, this function uses `fixate_field_nearest_int` to pick the value closest to the caps.
        """
        print("do_fixate_caps")
        if direction == Gst.PadDirection.SRC:
            return othercaps.fixate()
        else:
            in_info = GstVideo.VideoInfo()
            in_info.from_caps(caps)

            new_fps = in_info.fps_n / in_info.fps_d * self.vfi_factor
            new_fps_fraction = Fraction(new_fps)

            new_format = othercaps.get_structure(0).copy()
            new_format.fixate_field_nearest_fraction(
                "framerate", new_fps_fraction.numerator, new_fps_fraction.denominator
            )
            new_format.fixate_field_nearest_int("width", self.width)
            new_format.fixate_field_nearest_int("height", self.height)
            new_caps = Gst.Caps.new_empty()
            new_caps.append_structure(new_format)

            return new_caps.fixate()

    def do_generate_output(self) -> tuple[Gst.FlowReturn, Gst.Buffer | None]:
        """
        This virtual method allows producing 0 to N output buffers per input buffer.

        When a new buffer is chained on the sink pad, do_generate_output is called repeatedly
        as long as it returns Gst.FlowReturn.OK and a buffer.
        """
        print("do_generate_output")
        inbuf = self.queued_buf

        if self.frame_buffer:
            frame = self.frame_buffer.popleft()
            ret, outbuf = GstBase.BaseTransform.do_prepare_output_buffer(self, inbuf)
            outbuf.fill(0, frame)
            return ret, outbuf
        elif self.should_get_next_frame:
            self.should_get_next_frame = False
            return Gst.FlowReturn.OK, None
        else:
            _, outbuf = GstBase.BaseTransform.do_prepare_output_buffer(self, inbuf)
            ret = self.do_transform(inbuf, outbuf)
            self.should_get_next_frame = True
            return ret, outbuf

    def do_transform(
        self, inbuffer: Gst.Buffer, outbuffer: Gst.Buffer
    ) -> Gst.FlowReturn:
        print("do_transform")
        try:
            inbuf_info = inbuffer.map(Gst.MapFlags.READ)
            with inbuf_info:
                image_array = np.ndarray(
                    (self.height, self.width, self.pixel_bytes),
                    dtype=np.uint8,
                    buffer=inbuf_info.data,
                )
                frame = self.preprocess(image_array[:, :, :3])  # RGBA -> RGB
                vfi_frames = self._vfi_model.do_vfi(frame)

                for vfi_frame in vfi_frames:
                    self.frame_buffer.append(vfi_frame.cpu().numpy())

                # Return the original frame untouched
                outbuffer.fill(0, image_array)

                # Convert frames to Gst.Buffer
                # vfi_frame_bytes = bytes(
                #     vfi_frame.view(torch.uint8)
                # )  # Another approach to bytes: tensor.numpy().tobytes()
                # vfi_buffer = Gst.Buffer.new_wrapped(
                #     vfi_frame_bytes
                # )  # REVIEW: Possible missing argument?
                # vfi_buffer.pts = self.next_time
                # vfi_buffer.duration = self.frame_duration
                # self.next_time += self.frame_duration
                # buffer_list.insert(-1, vfi_buffer)
            return Gst.FlowReturn.OK
        except Gst.MapError as e:
            Gst.error(f"mapping error {e}")
            return Gst.FlowReturn.ERROR
        except Exception as e:
            Gst.error(f"{e}")
            return Gst.FlowReturn.ERROR


# GObject.type_register(VFIFilter)
__gstelementfactory__ = ("vfifilter", Gst.Rank.NONE, VFIFilter)
