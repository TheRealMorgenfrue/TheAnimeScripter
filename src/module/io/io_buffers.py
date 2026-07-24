import logging
import os
import queue
import subprocess
import threading
import time
import traceback
import types
from abc import abstractmethod
from queue import Queue
from typing import Any, override

import cv2
import nelux
import torch
from applib import LoggingManager
from torch import Tensor, dtype
from torch.nn import functional

from src.module.config.input_metadata_config import InputMetadataConfig
from src.module.config.tas_config import TASConfig
from src.module.io.encoding_settings import get_pix_fmt, match_encoder

from ..utils.cuda_checker import CudaChecker


class PeekQueue(Queue):
    """Wrapper class around queue.Queue to allow peeking at the next item in the queue without removing it."""

    @classmethod
    def __class_getitem__(cls, item: Any) -> types.GenericAlias:
        return super().__class_getitem__(item)

    def peek(self):
        """Peeks at the next item in the queue without removing it.

        Returns
        -------
            The next item in the queue.

        Raises
        ------
        queue.Empty
            If no next item is available.
        """
        with self.mutex:
            if len(self.queue) > 0:
                return self.queue[0]
        raise queue.Empty


class ReadBuffer:
    def __init__(
        self,
        input_path: str,
        width: int,
        height: int,
    ):
        """Creates a video decode buffer.

        Parameters
        ----------
        video_input : str
            Path to the input video file.
        width : int
            Width of the output frames.
        height : int
            Height of the output frames.

        Note
        ----
        NeLux returns HWC format [H, W, 3] with native dtype (uint8/int16).
        The method `_convert_frame_format` converts this to BCHW float format for processing.
        """
        self._logger = LoggingManager()
        self._frame_available = threading.Event()
        self._config = TASConfig()

        self.input_path = input_path
        self.width = width
        self.height = height
        self.decode_method: str = self._config["decode_method"]
        self.precision: str = self._config["precision"]
        self.bit_depth: str = self._config["bit_depth"]
        self.inpoint: float = self._config["inpoint"]
        self.outpoint: float = self._config["outpoint"]

        self.is_finished = False
        self.decode_buffer: Queue[Tensor | None] = Queue(maxsize=64)
        self.device_type = "cpu"
        self.cuda_norm_stream: torch.cuda.Stream | None = None

        if CudaChecker().cuda_available:
            try:
                self.cuda_norm_stream = torch.cuda.Stream()
                self.device_type = "cuda"
            except Exception:
                self._logger.error(
                    f"CUDA stream initialization failed, falling back to CPU.\n{traceback.format_exc()}"
                )

    def __call__(self):
        """Decodes frames from the video and stores them in the decodeBuffer."""
        decoded_frames = 0
        try:
            decoded_frames += self._decode_with_nelux()
        except Exception:
            self._logger.error(f"NeLux decoding error:\n{traceback.format_exc()}")

            if self.decode_method != "cpu":
                self._logger.warning(
                    "NeLux decode failed with non-CPU method; retrying with cpu."
                )
                self.decode_method = "cpu"
                try:
                    decoded_frames += self._decode_with_nelux()
                    return
                except Exception:
                    self._logger.error(
                        f"NeLux CPU retry failed:\n{traceback.format_exc()}"
                    )

            self._logger.info("Attempting fallback to OpenCV decoder...")
            try:
                decoded_frames += self._decode_with_opencv()
            except Exception:
                self._logger.error(f"OpenCV fallback failed:\n{traceback.format_exc()}")
        finally:
            self.decode_buffer.put(None)
            self._frame_available.set()

            self.is_finished = True
            self._logger.info(f"Decoded {decoded_frames} frames")

    def _decode_with_nelux(self) -> int:
        """Returns the number of frames decoded."""
        self._logger.info(
            f"Initializing new VideoReader for {self.input_path} ({self.decode_method})"
        )

        reader = nelux.VideoReader(
            self.input_path,
            decode_accelerator=self.decode_method,
            backend="pytorch",
        )

        if self.inpoint > 0 or self.outpoint > 0:
            reader[self.inpoint, self.outpoint]

        decoded_frames = 0

        for frame in reader:
            frame = self._convert_frame_format(frame, self.cuda_norm_stream)  # type: ignore
            self.decode_buffer.put(frame)
            self._frame_available.set()
            decoded_frames += 1
        return decoded_frames

    def _decode_with_opencv(self) -> int:
        """Returns the number of frames decoded."""
        self._logger.info(f"Initializing OpenCV VideoCapture for {self.input_path}")

        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video with OpenCV: {self.input_path}")

        decoded_frames = 0
        try:
            if self.inpoint > 0:
                cap.set(cv2.CAP_PROP_POS_MSEC, self.inpoint * 1000.0)

            while True:
                is_frame, frame = cap.read()
                if not is_frame:
                    break

                pts_millis = cap.get(cv2.CAP_PROP_POS_MSEC)
                pts_seconds = (
                    (pts_millis / 1000.0) if pts_millis and pts_millis > 0 else None
                )

                if (
                    pts_seconds is not None
                    and self.outpoint > 0
                    and pts_seconds >= self.outpoint
                ):
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self._convert_frame_format(
                    torch.from_numpy(frame), self.cuda_norm_stream
                )
                self.decode_buffer.put(frame)
                self._frame_available.set()
                decoded_frames += 1
        finally:
            cap.release()
        return decoded_frames

    def _convert_frame_format(
        self,
        frame: Tensor,
        norm_stream: torch.cuda.Stream | None = None,
        to_nhwc: bool = True,
    ) -> Tensor:
        """Converts a frame tensor from HWC to NCHW memory layout.

        While all PyTorch operations expect tensors to be in NCHW layout, NVIDIA tensor cores operate more efficiently with NHWC layout.
        Thus, depending on the workload, NHWC may be faster than NHWC.
        However, if the selected layout is not supported by a given tensor operation, necessary layout transformations will be applied to the tensor automatically.

        A brief guide regarding memory layouts is available [here](https://uxlfoundation.github.io/oneDNN/dev_guide_understanding_memory_formats.html).

        Parameters
        ----------
        frame : Tensor
            A frame in NHWC format.
        norm_stream : torch.cuda.Stream | None, optional
            The CUDA stream for normalization, by default None.

        Returns
        -------
        Tensor
            The frame converted to NCHW layout.
        """
        with torch.cuda.stream(norm_stream):
            try:
                frame = frame.pin_memory()
            except Exception:
                pass

            if to_nhwc:
                # TODO: Test if this works
                frame = frame.to(
                    device=self.device_type,
                    non_blocking=norm_stream is not None,
                    dtype=torch.float16 if self.precision else torch.float32,
                    memory_format=torch.channels_last,  # NHWC, more efficient on Tensor Cores https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#prefer_nhwc
                )
            else:
                frame = frame.to(
                    device=self.device_type,
                    non_blocking=norm_stream is not None,
                    dtype=torch.float16 if self.precision else torch.float32,
                )

                norm = 1 / 255.0 if frame.dtype == torch.uint8 else 1 / 65535.0
                frame = frame.permute(2, 0, 1).mul(norm).clamp(0, 1).unsqueeze(0)

        if norm_stream is not None:
            norm_stream.synchronize()

        return frame

    def read(self) -> Tensor | None:
        """Reads a frame from the decodeBuffer.

        Returns
        -------
        Tensor
            The next frame from the decodeBuffer.
        """
        return self.decode_buffer.get()

    def peek(self) -> Tensor | None:
        """Peeks at the next frame in the decodeBuffer without removing it.

        Returns
        -------
        Tensor | None
            The next frame from the decodeBuffer, or None if decoding is finished and the queue is empty.
        """
        while True:
            if self.is_finished:
                return None

            with self.decode_buffer.mutex:
                if len(self.decode_buffer.queue) > 0:
                    return self.decode_buffer.queue[0]

            self._frame_available.wait(timeout=0.1)
            self._frame_available.clear()

    def is_read_finished(self) -> bool:
        """Returns True if the decoding process is finished."""
        return self.is_finished

    def is_queue_empty(self) -> bool:
        """Returns True if the decoding buffer is empty and the decoding process is finished."""
        return self.decode_buffer.empty() and self.is_finished


class WriteBuffer:
    def __init__(self) -> None:
        self._logger = LoggingManager()
        self._config = TASConfig()

        self.write_buffer: Queue[Tensor | None] = Queue(maxsize=64)

    @abstractmethod
    def __call__(self):
        """Process frames from `self.write_buffer` and encode to destination file."""
        ...

    def write(self, frame: Tensor):
        """Add a frame to the write buffer.

        Parameters
        ----------
        frame : Tensor
            The frame to add. Must be in BCHW format.
        """
        self.write_buffer.put(frame)

    def close(self):
        """Close the write stream."""
        self.write_buffer.put(None)


class WriteBufferFFmpeg(WriteBuffer):
    def __init__(
        self,
        input_path: str,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        encode_method: str,
        input_metadata_config: InputMetadataConfig,
        grayscale: bool = False,
        transparent: bool = False,
    ):
        """A class meant to pipe the input to FFmpeg from a queue.

        Parameters
        ----------
        input : str
            The path to the input file.
        output : str
            The path to the output file.
        width : int
            The width of the output video in pixels.
        height : int
            The height of the output video in pixels.
        fps : float
            The framerate of the output video.
        encode_method : str
            The encoding method to encode the output file with.
        input_metadata : InputMetadataConfig
            The metadata of the input file.
        grayscale : bool, optional
            Whether to encode the video in grayscale, by default False.
        transparent : bool, optional
            Whether to encode the video with transparency, by default False.
        """
        super().__init__()
        self._input_metadata_config = input_metadata_config

        self.input_path = input_path
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.encode_method = encode_method
        self.grayscale = grayscale
        self.transparent = transparent
        self.inpoint: float = self._config["inpoint"]
        self.outpoint: float = self._config["outpoint"]
        self.custom_encoder: str = self._config["custom_encoder"]
        self.bit_depth: str = self._config["bit_depth"]

    def _encode_settings(self) -> list[str]:
        """
        Simplified structure for setting input/output pix formats
        and building FFmpeg command.
        """
        # Set environment variables
        os.environ["FFREPORT"] = "file=FFmpeg-Log.log:level=32"
        if "av1" in [self.encode_method, self.custom_encoder]:
            os.environ["SVT_LOG"] = "0"

        if self.encode_method == "png" and "%" not in self.output_path:
            _, ext = os.path.splitext(self.output_path)
            if not ext:
                self.output_path = os.path.join(self.output_path, "%08d.png")
            else:
                base, _ = os.path.splitext(self.output_path)
                self.output_path = f"{base}_%08d.png"

        input_pix_fmt, output_pix_fmt, self.encode_method = get_pix_fmt(
            self.encode_method, self.bit_depth, self.grayscale, self.transparent
        )

        if self._config["benchmark"]:
            return self._build_benchmark_command(input_pix_fmt)
        else:
            return self._build_encoding_command(input_pix_fmt, output_pix_fmt)

    def _build_benchmark_command(self, input_pix_fmt: str) -> list[str]:
        """Build FFmpeg command for benchmarking."""
        return [
            self._config["ffmpeg"],
            "-y",
            "-hide_banner",
            "-v",
            "warning",
            "-nostats",
            "-f",
            "rawvideo",
            "-video_size",
            f"{self.width}x{self.height}",
            "-pix_fmt",
            input_pix_fmt,
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-benchmark",
            "-f",
            "null",
            "-",
        ]

    def _build_encoding_command(
        self, input_pix_fmt: str, output_pix_fmt: str
    ) -> list[str]:
        """Build FFmpeg command for encoding."""
        use_hardware_accel = "nvenc" in self.encode_method and not self.custom_encoder
        use_audio_subs = self._config["audio_subs"]

        command = [
            self._config["ffmpeg"],
            "-y",
            "-hide_banner",
            "-loglevel",
            "quiet",
            "-nostats",
            "-threads",
            "0",
            "-filter_threads",
            "0",
        ]

        # Initialize CUDA device when using NVENC
        if use_hardware_accel:
            command.extend(["-init_hw_device", "cuda=cu:0", "-filter_hw_device", "cu"])

        command.extend(
            [
                "-f",
                "rawvideo",
                "-pix_fmt",
                input_pix_fmt,
                "-s",
                f"{self.width}x{self.height}",
                "-r",
                str(self.fps),
            ]
        )

        if self.outpoint != 0 and not self._config["slowmo"]:
            command.extend(
                [
                    "-itsoffset",
                    str(self.inpoint),
                    "-i",
                    "pipe:0",
                    "-ss",
                    str(self.inpoint),
                    "-to",
                    str(self.outpoint),
                ]
            )
        else:
            command.extend(["-i", "pipe:0"])

        if use_audio_subs:
            command.extend(["-thread_queue_size", "1024", "-i", self.input_path])

        filter_list = self._build_filter_list(input_pix_fmt)

        command.extend(["-map", "0:v"])

        if self.custom_encoder:
            command.extend(self._build_custom_encoder(filter_list, output_pix_fmt))
        else:
            command.extend(match_encoder(self.encode_method))

            if use_hardware_accel:
                filter_list.append("format=nv12")
                filter_list.append("hwupload_cuda")
                command.extend(["-vf", ",".join(filter_list)])
            else:
                # FIXME: Check if "output_pix_fmt" is meant not to be applied when "use_hardware_accel == True"
                if filter_list:
                    command.extend(["-vf", ",".join(filter_list)])
                command.extend(["-pix_fmt", output_pix_fmt])

        if use_audio_subs:
            command.extend(self._build_audio_settings())

        command.append(self.output_path)
        return command

    def _build_filter_list(self, input_pix_fmt: str) -> list[str]:
        """Build list of video filters based on settings"""
        filter_list: list[str] = []

        if self.grayscale:
            filter_list.append(
                "format=gray" if self.bit_depth == "8bit" else "format=gray16be"
            )
        if self.transparent:
            filter_list.append("format=yuva420p")

        if not self.grayscale and not self.transparent:
            color_space_filter = {
                "bt709": f"zscale=matrix=709:dither=error_diffusion,format={input_pix_fmt}",
                "bt2020": "zscale=matrix=bt2020:norm=bt2020:dither=error_diffusion,format=yuv420p",
            }

            color_space_fields = ["color_primaries", "color_transfer"]
            detected_color_space = None

            for color_field in color_space_fields:
                color_space: str = self._input_metadata_config.get_value(
                    color_field, default=""
                )
                if color_space in color_space_filter:
                    detected_color_space = color_space
                    break
            else:
                # Fallback color space
                detected_color_space = "bt709"
            filter_list.append(color_space_filter[detected_color_space])
        return filter_list

    def _build_custom_encoder(
        self, filter_list: list[str], output_pix_fmt: str
    ) -> list[str]:
        """Apply custom encoder settings with filters"""
        custom_encoder_args = self.custom_encoder.split()

        if "-vf" in custom_encoder_args:
            vf_index = custom_encoder_args.index("-vf")
            filter_string = custom_encoder_args[vf_index + 1]
            for filter_item in filter_list:
                filter_string += f",{filter_item}"
            custom_encoder_args[vf_index + 1] = filter_string
        elif filter_list:
            custom_encoder_args.extend(["-vf", ",".join(filter_list)])

        if "-pix_fmt" not in custom_encoder_args:
            self._logger.info(f"-pix_fmt was not found, adding {output_pix_fmt}.")
            custom_encoder_args.extend(["-pix_fmt", output_pix_fmt])

        return custom_encoder_args

    def _build_audio_settings(self) -> list[str]:
        """Build audio encoding settings"""
        audio_settings = ["-map", "1:a"]

        # TODO: include attachments and chapters
        audio_codec = "copy"
        sub_codec = "copy"
        if self.output_path.endswith(".webm"):
            audio_codec = "libopus"
            sub_codec = "webvtt"
        audio_settings.extend(["-c:a", audio_codec, "-map", "1:s?", "-c:s", sub_codec])

        if self.outpoint != 0:
            audio_settings.extend(["-ss", str(self.inpoint), "-to", str(self.outpoint)])

        return audio_settings

    def _process_frame(
        self, frame: Tensor, multiplier: int, dtype: dtype, needs_resize: bool
    ) -> Tensor:
        if needs_resize:
            frame = functional.interpolate(
                frame,
                size=(self.height, self.width),
                mode="bicubic",
                align_corners=False,
            )

        return (
            frame.squeeze(0)
            .permute(1, 2, 0)
            .mul(multiplier)
            .clamp(0, multiplier)
            .to(dtype)
            .contiguous()
        )

    @override
    def __call__(self):
        written_frames = 0

        # Wait for at least one frame to be queued before starting encoding
        while self.write_buffer.empty():
            try:
                time.sleep(0.001)
            except KeyboardInterrupt:
                logging.warning("Encoding interrupted by user")
                return

        ffmpeg_proc = None
        try:
            initial_frame = self.write_buffer.queue[0]

            self.channels = 1 if self.grayscale else 4 if self.transparent else 3

            if self.bit_depth == "8bit":
                multiplier = 255
                dtype = torch.uint8
            else:
                multiplier = 65535
                dtype = torch.uint16

            needs_resize = (
                initial_frame.shape[2] != self.height
                or initial_frame.shape[3] != self.width
            )

            if needs_resize:
                self._logger.warning(
                    f"Frame size mismatch. Frame: {initial_frame.shape[3]}x{initial_frame.shape[2]}, Output: {self.width}x{self.height}"
                )

            command = self._encode_settings()
            self._logger.debug(f"Encode command: {' '.join(map(str, command))}")

            use_cuda = False
            transfer_stream = None
            if CudaChecker().cuda_available:
                try:
                    transfer_stream = torch.cuda.Stream()
                    use_cuda = True
                except Exception as e:
                    self._logger.error(
                        f"CUDA init failed in writer, using CPU path. Reason: {e}"
                    )
                    use_cuda = False

            ffmpeg_proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=None,
                stderr=subprocess.DEVNULL,
                shell=False,
            )

            if use_cuda:
                frame_shape = (self.height, self.width, self.channels)
                pinnedBuffers = [
                    torch.empty(frame_shape, dtype=dtype, pin_memory=True),
                    torch.empty(frame_shape, dtype=dtype, pin_memory=True),
                ]
                transfer_events = [torch.cuda.Event(), torch.cuda.Event()]
                buffer_index = 0
                pending_buffer = None
                pending_event = None

                while True:
                    try:
                        frame = self.write_buffer.get(timeout=1.0)
                    except Exception:
                        time.sleep(0.001)
                        continue

                    if pending_buffer is not None and pending_event is not None:
                        pending_event.synchronize()
                        ffmpeg_proc.stdin.write(memoryview(pending_buffer.numpy()))  # type: ignore
                        written_frames += 1

                    if frame is not None:
                        with torch.cuda.stream(transfer_stream):
                            out_frame = self._process_frame(
                                frame, multiplier, dtype, needs_resize
                            )
                            current_buffer = pinnedBuffers[buffer_index]
                            current_buffer.copy_(out_frame, non_blocking=True)
                            current_event = transfer_events[buffer_index]
                            current_event.record(transfer_stream)  # type: ignore
                            pending_buffer = current_buffer
                            pending_event = current_event
                            buffer_index = 1 - buffer_index
            else:
                while True:
                    try:
                        frame = self.write_buffer.get(timeout=1.0)
                    except Exception:
                        time.sleep(0.001)
                        continue

                    if frame is None:
                        break

                    out_frame = self._process_frame(
                        frame, multiplier, dtype, needs_resize
                    )
                    ffmpeg_proc.stdin.write(memoryview(out_frame.numpy()))  # type: ignore
                    written_frames += 1
            self._logger.debug(f"Encoded {written_frames} frames")
        except Exception:
            self._logger.error(f"Encoding error:\n{traceback.format_exc()}")
        finally:
            try:
                if ffmpeg_proc is not None and ffmpeg_proc.stdin:
                    ffmpeg_proc.stdin.close()
                if ffmpeg_proc is not None:
                    ffmpeg_proc.wait(timeout=3)
            except Exception:
                self._logger.error(f"Cleanup error:\n{traceback.format_exc()}")


class WriteBufferNeLux(WriteBuffer):
    """
    Write buffer that uses NeLux VideoEncoder for NVENC encoding.
    More efficient than FFmpeg pipe for GPU-resident frames.
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        encode_method: str,
        **kwargs,  # Accept and ignore other WriteBuffer params for compatibility
    ):
        """Initialize the NeLux-based encoder.

        Parameters
        ----------
        input_path : str
            The path to the input file.
        output_path : str
            The path to the output file.
        width : int
            The width of the output video in pixels.
        height : int
            The height of the output video in pixels.
        fps : float
            The framerate of the output video.
        encode_method : str
            The encoding method to encode the output file with.
        """
        super().__init__()
        self.input_path = input_path
        self.output_path = os.path.normpath(output_path)
        self.width = width
        self.height = height
        self.fps = fps
        self.inpoint: float = self._config["inpoint"]
        self.outpoint: float = self._config["outpoint"]

        codec_map = {
            "nvenc_h264_nelux": "h264_nvenc",
            "nvenc_h265_nelux": "hevc_nvenc",
            "nvenc_av1_nelux": "av1_nvenc",
        }
        self.codec = codec_map.get(encode_method, "h264_nvenc")
        self.encoder = None

        if CudaChecker.cuda_available:
            # An exception here should propagate to caller
            self.cuda_stream = torch.cuda.Stream()

        self._logger.info(
            f"NeLux write buffer initialized: {width}x{height}@{fps}fps, codec={self.codec}"
        )

    @override
    def __call__(self):
        written_frames = 0
        try:
            while self.write_buffer.empty():
                time.sleep(0.001)

            self.encoder = nelux.VideoEncoder(
                self.output_path,
                codec=self.codec,
                width=self.width,
                height=self.height,
                fps=self.fps,
            )

            if hasattr(self.encoder, "is_hardware_encoder"):
                if self.encoder.is_hardware_encoder:
                    self._logger.debug(
                        f"NeLux NVENC encoder confirmed: {self.codec} -> {self.output_path}"
                    )
                else:
                    self._logger.warning(
                        f"NeLux encoder is NOT using hardware NVENC! Codec: {self.codec}"
                    )
            else:
                self._logger.debug(
                    f"Nelux encoder created: {self.codec} -> {self.output_path}"
                )

            while True:
                try:
                    frame = self.write_buffer.get(timeout=1.0)
                except Exception:
                    time.sleep(0.001)
                    continue

                if frame is None:
                    break

                with torch.cuda.stream(self.cuda_stream):
                    frame = frame.squeeze(0).permute(1, 2, 0)
                    frame = (
                        frame.mul(255.0)
                        .clamp(0, 255)
                        .to(dtype=torch.uint8, non_blocking=True)
                        .contiguous()
                    )
                self.cuda_stream.synchronize()
                self.encoder.encode_frame(frame)
                written_frames += 1

            self._logger.debug(f"NeLux encoded {written_frames} frames")
        except Exception:
            self._logger.error(f"NeLux encoding error:\n{traceback.format_exc()}")
        finally:
            if self.encoder is not None:
                try:
                    self.encoder.close()
                except Exception:
                    self._logger.error(
                        f"Error closing NeLux encoder:\n{traceback.format_exc()}"
                    )


def create_write_buffer(encode_method: str, **kwargs) -> WriteBuffer:
    """Factory function to create the appropriate write buffer.

    Parameters
    ----------
    encode_method : str
        The encoding method
    **kwargs: Arguments passed to the buffer constructor.

    Returns
    -------
    WriteBuffer
        The write buffer instance.

    Example Usage
    -------------
    buffer = createWriteBuffer(
        encode_method=args.encode_method,
        input=args.input,
        output=args.output,
        width=width,
        height=height,
        fps=fps,
        ...
    )
    """
    if encode_method.endswith("_nelux"):
        return WriteBufferNeLux(encode_method=encode_method, **kwargs)
    else:
        return WriteBufferFFmpeg(encode_method=encode_method, **kwargs)
