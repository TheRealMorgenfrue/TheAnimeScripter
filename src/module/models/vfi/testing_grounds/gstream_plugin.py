import random
import time

import gi
import numpy

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
gi.require_version("GstAnalytics", "1.0")

from gi.repository import GLib, GObject, Gst, GstAnalytics, GstBase  # noqa: E402


def ML_magic(frame):
    """
    Dummy function to emulate ML process that takes a low resolution frame as input
    and returns a region of interest center point.
    """
    # Always return a dummy point (e.g., center of the resized frame)
    x = random.randint(0, frame.shape[1])
    y = random.randint(0, frame.shape[0])
    return x, y


Gst.init(None)
Gst.init_python()


SRC_CAPS = Gst.Caps(
    Gst.Structure(
        "video/x-raw",
        format="RGB",
        width=Gst.IntRange(range(1, GLib.MAXINT)),
        height=Gst.IntRange(range(1, GLib.MAXINT)),
        framerate=Gst.FractionRange(Gst.Fraction(1, 1), Gst.Fraction(GLib.MAXINT, 1)),
    )
)

SINK_CAPS = Gst.Caps(
    Gst.Structure(
        "video/x-raw",
        format="RGB",
        width=Gst.IntRange(range(1, GLib.MAXINT)),
        height=Gst.IntRange(range(1, GLib.MAXINT)),
        framerate=Gst.FractionRange(Gst.Fraction(1, 1), Gst.Fraction(GLib.MAXINT, 1)),
    )
)

SRC_PAD_TEMPLATE = Gst.PadTemplate.new(
    "src", Gst.PadDirection.SRC, Gst.PadPresence.ALWAYS, SRC_CAPS
)

SINK_PAD_TEMPLATE = Gst.PadTemplate.new(
    "sink", Gst.PadDirection.SINK, Gst.PadPresence.ALWAYS, SINK_CAPS
)


class SampleFilter(GstBase.BaseTransform):
    __gstmetadata__ = (
        "OpenCV Filter",
        "Filter",
        "OpenCV Sample Filter",
        "Ruben Gonzaelz <rgonzalez@fluendo.com>",
    )

    __gsttemplates__ = (SRC_PAD_TEMPLATE, SINK_PAD_TEMPLATE)

    def __init__(self):
        super().__init__()
        self.set_qos_enabled(True)

    def do_set_caps(self, incaps, outcaps):
        s = incaps.get_structure(0)
        self.width = s.get_int("width").value
        self.height = s.get_int("height").value

        return True

    def do_transform_ip(self, inbuf):
        try:
            inbuf_info = inbuf.map(Gst.MapFlags.READ)
            with inbuf_info:
                frame = numpy.ndarray(
                    shape=(self.height, self.width, 3),
                    dtype=numpy.uint8,
                    buffer=inbuf_info.data,
                )

                # Call the ML_magic function
                x, y = ML_magic(frame)

            meta = GstAnalytics.buffer_add_analytics_relation_meta(inbuf)
            label = GLib.quark_from_string("label")
            meta.add_od_mtd(label, x, y, 20, 20, 0.55)

            return Gst.FlowReturn.OK

        except Gst.MapError as e:
            Gst.error("mapping error %s" % e)
            return Gst.FlowReturn.ERROR
        except Exception as e:
            Gst.error("%s" % e)
            return Gst.FlowReturn.ERROR


GObject.type_register(SampleFilter)
Gst.Element.register(None, "sample_filter", Gst.Rank.NONE, SampleFilter)

gstreamer_pipeline = """
                videotestsrc num-buffers=100 ! video/x-raw, width=640, height=480, framerate=30/1 ! videoconvert ! video/x-raw, width=640, height=480, framerate=30/1 !
                originalbuffersave ! videoconvertscale ! video/x-raw, width=100, height=100 ! sample_filter ! originalbufferrestore ! video/x-raw, width=640, height=480, framerate=30/1 ! videoconvert ! objectdetectionoverlay ! tee name=t !
                queue ! videoconvert !  x264enc tune=zerolatency bitrate=2000 speed-preset=ultrafast !  flvmux streamable=true ! udpsink sync=false
                t. ! queue ! videoconvert ! xvimagesink handle-events=false sync=true
"""


# GStreamer pipeline
pipeline = Gst.parse_launch(gstreamer_pipeline)

# Start processing
start_time = time.time()
pipeline.set_state(Gst.State.PLAYING)
bus = pipeline.get_bus()

# Wait for the pipeline to finish
msg = bus.timed_pop_filtered(
    Gst.CLOCK_TIME_NONE, Gst.MessageType.ERROR | Gst.MessageType.EOS
)

if msg:
    t = msg.type
    if t == Gst.MessageType.ERROR:
        err, debug = msg.parse_error()
        print(f"Error: {err}, {debug}")
    elif t == Gst.MessageType.EOS:
        print("Pipeline finished successfully.")

pipeline.set_state(Gst.State.NULL)

# Log total video processing time
total_time = time.time() - start_time
print(f"Total wall-clock processing time for the video: {total_time:.2f} seconds")
