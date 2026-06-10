import logging
import os
import subprocess
import threading
import time
import traceback
from enum import Enum
from queue import Queue

import cv2
import nelux
import torch
from applib import LoggingManager
from numpy.typing import NDArray
from torch import Tensor
from torch.nn import functional

from module.config.tas_config import TASConfig
from src.module.utils.encodingSettings import getPixFMT, matchEncoder

from .cuda_checker import CudaChecker


class Backend(Enum):
    TORCH = "pytorch"
    NUMPY = "numpy"


class ReadBuffer:
    def __init__(
        self,
        video_input: str = "",
        inpoint: float = 0.0,
        outpoint: float = 0.0,
        half: bool = True,
        resize: bool = False,
        width: int = 1920,
        height: int = 1080,
        bit_depth: str = "8bit",
        backend: Backend = Backend.TORCH,
        decode_method: str = "cpu",
        batched: bool = False,
        batch_size: int = 1,
    ):
        """
        Creates a video decode buffer.

        Args:
            videoInput (str): Path to the input video file.
            inpoint (float): Start time of the segment to decode, in seconds.
            outpoint (float): End time of the segment to decode, in seconds.
            half (bool): Whether to use half precision (float16) for tensors.
            resize (bool): Whether to resize the frames.
            width (int): Width of the output frames.
            height (int): Height of the output frames.
            bitDepth (str): Bit depth of the output frames, e.g., "8bit" or "10bit".
            toTorch (bool): Whether to convert frames to torch tensors.
            decode_method (str): The backend to use for decoding, e.g., "cpu" or "nvdec".
            batched (bool): Whether to decode frames in batches.
            batchSize (int): The size of each batch when decoding in batches.

        Note:
            NeLux returns HWC format [H, W, 3] with native dtype (uint8/int16).
            `process_frame_to_torch` converts this to BCHW float format for processing.
        """
        self._logger = LoggingManager()
        self._frame_available = threading.Event()
        self._config = TASConfig()
        self._checker = CudaChecker()

        self.decode_method = decode_method
        self.half = half
        self.width = width
        self.height = height
        self.resize = resize
        self.bit_depth = bit_depth
        self.video_input = os.path.normpath(video_input)
        self.inpoint = inpoint
        self.outpoint = outpoint

        self.is_finished = False
        self.decode_buffer: Queue[Tensor | NDArray] = Queue(maxsize=64)
        self.device_type = "cpu"
        self.backend = backend
        self.cuda_norm_stream: torch.cuda.Stream | None = None

        if self._checker.cuda_available and self.backend == Backend.TORCH:
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
            # self.decode_buffer.put(None)
            # self._frame_available.set()

            self.is_finished = True
            self._logger.info(f"Decoded {decoded_frames} frames")

    def _decode_with_nelux(self) -> int:
        """Returns the number of frames decoded."""
        self._logger.info(
            f"Initializing new VideoReader for {self.video_input} ({self.decode_method})"
        )

        reader = nelux.VideoReader(
            self.video_input,
            decode_accelerator=self.decode_method,
            backend=self.backend.value,
        )

        if self.inpoint > 0 or self.outpoint > 0:
            reader[self.inpoint, self.outpoint]

        decoded_frames = 0

        match self.backend:
            case Backend.TORCH:
                for frame in reader:
                    frame = self.convert_frame_format(frame, self.cuda_norm_stream)  # type: ignore
                    self.decode_buffer.put(frame)
                    self._frame_available.set()
                    decoded_frames += 1
            case Backend.NUMPY:
                for frame in reader:
                    self.decode_buffer.put(frame)
                    self._frame_available.set()
                    decoded_frames += 1
        return decoded_frames

    def _decode_with_opencv(self) -> int:
        """Returns the number of frames decoded."""

        def opencv_decode_helper(cap: cv2.VideoCapture) -> Tensor | None:
            ret, frame = cap.read()
            if not ret:
                return None

            pts_millis = cap.get(cv2.CAP_PROP_POS_MSEC)
            pts_seconds = (
                (pts_millis / 1000.0) if pts_millis and pts_millis > 0 else None
            )

            if pts_seconds is not None and end_time > 0 and pts_seconds >= end_time:
                return None

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return torch.from_numpy(frame)

        self._logger.info(f"Initializing OpenCV VideoCapture for {self.video_input}")

        cap = cv2.VideoCapture(self.video_input)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video with OpenCV: {self.video_input}")

        decoded_frames = 0
        start_time = self.inpoint
        end_time = self.outpoint

        try:
            if start_time > 0:
                cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000.0)

            match self.backend:
                case Backend.TORCH:
                    while True:
                        frame = opencv_decode_helper(cap)
                        if frame is None:
                            break
                        frame = self.convert_frame_format(frame, self.cuda_norm_stream)
                        self.decode_buffer.put(frame)
                        self._frame_available.set()
                        decoded_frames += 1
                case Backend.NUMPY:
                    while True:
                        frame = opencv_decode_helper(cap)
                        if frame is None:
                            break
                        frame = frame.numpy()
                        self.decode_buffer.put(frame)
                        self._frame_available.set()
                        decoded_frames += 1
        finally:
            cap.release()

        return decoded_frames

    def convert_frame_format(
        self,
        frame: Tensor,
        norm_stream: torch.cuda.Stream | None = None,
    ) -> Tensor:
        """Converts a single frame with optimized memory handling.

        Parameters
        ----------
        frame : Tensor
            A frame in NeLux format (HWC format)
        norm_stream : torch.cuda.Stream | None, optional
            The CUDA stream for normalization, by default None.

        Returns
        -------
        Tensor
            The frame converted to BCHW format.
        """
        norm = 1 / 255.0 if frame.dtype == torch.uint8 else 1 / 65535.0

        with torch.cuda.stream(norm_stream):
            try:
                frame = frame.pin_memory()
            except Exception:
                pass

            frame = frame.to(
                device=self.device_type,
                non_blocking=norm_stream is not None,
                dtype=torch.float16 if self.half else torch.float32,
            )

            frame = frame.permute(2, 0, 1).mul(norm).clamp(0, 1)

            if self.resize:
                frame = functional.interpolate(
                    frame.unsqueeze(0),
                    size=(self.height, self.width),
                    mode="bicubic",
                    align_corners=False,
                )
            else:
                frame = frame.unsqueeze(0)

        if norm_stream is not None:
            norm_stream.synchronize()

        return frame

    def read(self) -> Tensor | NDArray:
        """Reads a frame from the decodeBuffer.

        Returns
        -------
        Tensor | NDArray
            The next frame from the decodeBuffer.
        """
        return self.decode_buffer.get()

    def peek(self) -> Tensor | NDArray | None:
        """Peeks at the next frame in the decodeBuffer without removing it.

        Returns
        -------
        Tensor | NDArray | None
            The next frame from the decodeBuffer, or None if decoding is finished and the queue is empty.
        """
        while True:
            with self.decode_buffer.mutex:
                if len(self.decode_buffer.queue) > 0:
                    return self.decode_buffer.queue[0]

            if self.is_finished:
                return None

            self._frame_available.wait(timeout=0.1)
            self._frame_available.clear()

    def is_read_finished(self) -> bool:
        """Returns True if the decoding process is finished."""
        return self.is_finished

    def is_queue_empty(self) -> bool:
        """Returns True if the decoding buffer is empty and the decoding process is finished."""
        return self.decode_buffer.empty() and self.is_finished


class WriteBuffer:
    def __init__(
        self,
        input: str = "",
        output: str = "",
        encode_method: str = "x264",
        custom_encoder: str = "",
        width: int = 1920,
        height: int = 1080,
        fps: float = 60.0,
        sharpen: bool = False,
        sharpen_sens: float = 0.0,
        grayscale: bool = False,
        transparent: bool = False,
        benchmark: bool = False,
        bitDepth: str = "8bit",
        inpoint: float = 0.0,
        outpoint: float = 0.0,
        slowmo: bool = False,
        output_scale_width: int = None,
        output_scale_height: int = None,
        enablePreview: bool = False,
    ):
        """
        A class meant to Pipe the input to FFMPEG from a queue.

        output: str - The path to the output video file.
        encode_method: str - The method to use for encoding the video. Options include "x264", "x264_animation", "nvenc_h264", etc.
        custom_encoder: str - A custom encoder string to use for encoding the video.
        grayscale: bool - Whether to encode the video in grayscale.
        width: int - The width of the output video in pixels.
        height: int - The height of the output video in pixels.
        fps: float - The frames per second of the output video.
        sharpen: bool - Whether to apply a sharpening filter to the video.
        sharpen_sens: float - The sensitivity of the sharpening filter.
        transparent: bool - Whether to encode the video with transparency.
        audio: bool - Whether to include audio in the output video.
        benchmark: bool - Whether to benchmark the encoding process, this will not output any video.
        bitDepth: str - The bit depth of the output video. Options include "8bit" and "10bit".
        inpoint: float - The start time of the segment to encode, in seconds.
        outpoint: float - The end time of the segment to encode, in seconds.
        output_scale_width: int - The target width for output scaling (optional).
        output_scale_height: int - The target height for output scaling (optional).
        enablePreview: bool - Whether to enable FFmpeg-based preview output (optional).
        """
        self.input = input
        self.output = os.path.normpath(output)
        self.encode_method = encode_method

        if self.encode_method == "png" and "%" not in self.output:
            _, ext = os.path.splitext(self.output)
            if not ext:
                self.output = os.path.join(self.output, "%08d.png")
            else:
                base, _ = os.path.splitext(self.output)
                self.output = f"{base}_%08d.png"

        self.custom_encoder = custom_encoder
        self.grayscale = grayscale
        self.width = width
        self.height = height
        self.fps = fps
        self.sharpen = sharpen
        self.sharpen_sens = sharpen_sens
        self.transparent = transparent
        self.benchmark = benchmark
        self.bitDepth = bitDepth
        self.inpoint = inpoint
        self.outpoint = outpoint
        self.slowmo = slowmo
        self.output_scale_width = output_scale_width
        self.output_scale_height = output_scale_height
        self.enablePreview = enablePreview

        self.writtenFrames = 0
        self.writeBuffer = Queue(maxsize=64)

        self.previewPath = (
            os.path.join(cs.WHEREAMIRUNFROM, "preview.jpg") if enablePreview else None
        )

    def encodeSettings(self) -> list:
        """
        Simplified structure for setting input/output pix formats
        and building FFMPEG command.
        """
        # Set environment variables
        os.environ["FFREPORT"] = "file=FFmpeg-Log.log:level=32"
        if "av1" in [self.encode_method, self.custom_encoder]:
            os.environ["SVT_LOG"] = "0"

        self.inputPixFmt, outputPixFmt, self.encode_method = getPixFMT(
            self.encode_method, self.bitDepth, self.grayscale, self.transparent
        )

        if self.benchmark:
            return self._buildBenchmarkCommand()
        else:
            return self._buildEncodingCommand(outputPixFmt)

    def _buildBenchmarkCommand(self):
        """Build FFmpeg command for benchmarking"""
        return [
            cs.FFMPEGPATH,
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
            self.inputPixFmt,
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-benchmark",
            "-f",
            "null",
            "-",
        ]

    def _isNvencEncoder(self):
        """Check if the current encode method uses NVENC"""
        nvenc_methods = [
            "nvenc_h264",
            "slow_nvenc_h264",
            "nvenc_h265",
            "slow_nvenc_h265",
            "nvenc_h265_10bit",
            "nvenc_av1",
            "slow_nvenc_av1",
            "lossless_nvenc_h264",
        ]
        return self.encode_method in nvenc_methods

    def _buildEncodingCommand(self, outputPixFmt):
        """Build FFmpeg command for encoding"""
        useHwUpload = self._isNvencEncoder() and not self.custom_encoder

        command = [
            cs.FFMPEGPATH,
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

        # Initialize CUDA device for hwupload when using NVENC
        if useHwUpload:
            command.extend(["-init_hw_device", "cuda=cu:0", "-filter_hw_device", "cu"])

        command.extend(
            [
                "-f",
                "rawvideo",
                "-pix_fmt",
                self.inputPixFmt,
                "-s",
                f"{self.width}x{self.height}",
                "-r",
                str(self.fps),
            ]
        )

        if self.outpoint != 0 and not self.slowmo:
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

        if cs.AUDIO:
            command.extend(["-thread_queue_size", "1024", "-i", self.input])

        filterList = self._buildFilterList()

        if self.enablePreview:
            filterComplexParts = []

            if filterList:
                baseFilters = ",".join(filterList)
                filterComplexParts.append(f"[0:v]{baseFilters},split=2[main][preview]")
            else:
                filterComplexParts.append("[0:v]split=2[main][preview]")

            filterComplexParts.append("[preview]fps=2[previewThrottled]")

            combinedFilter = ";".join(filterComplexParts)
            command.extend(
                ["-filter_complex", combinedFilter, "-filter_complex_threads", "0"]
            )

            command.extend(["-map", "[main]"])

            if not self.custom_encoder:
                command.extend(matchEncoder(self.encode_method))
                command.extend(["-pix_fmt", outputPixFmt])
            else:
                customArgs = self.custom_encoder.split()
                if "-vf" in customArgs:
                    vfIdx = customArgs.index("-vf")
                    customArgs.pop(vfIdx)
                    customArgs.pop(vfIdx)
                if "-pix_fmt" not in customArgs:
                    customArgs.extend(["-pix_fmt", outputPixFmt])
                command.extend(customArgs)

            if cs.AUDIO:
                command.extend(self._buildAudioSettings())

            command.append(self.output)

            command.extend(
                [
                    "-map",
                    "[previewThrottled]",
                    "-q:v",
                    "2",
                    "-update",
                    "1",
                    self.previewPath,
                ]
            )
        else:
            command.extend(["-map", "0:v"])

            if not self.custom_encoder:
                command.extend(matchEncoder(self.encode_method))

                if useHwUpload:
                    hwFilters = filterList.copy() if filterList else []
                    hwFilters.append("format=nv12")
                    hwFilters.append("hwupload_cuda")
                    command.extend(["-vf", ",".join(hwFilters)])
                else:
                    if filterList:
                        command.extend(["-vf", ",".join(filterList)])
                    command.extend(["-pix_fmt", outputPixFmt])
            else:
                command.extend(self._buildCustomEncoder(filterList, outputPixFmt))

            if cs.AUDIO:
                command.extend(self._buildAudioSettings())

            command.append(self.output)

        return command

    def _getOutputFormat(self):
        ext = os.path.splitext(self.output)[1].lower()
        formatMap = {
            ".mp4": "mp4",
            ".mkv": "matroska",
            ".webm": "webm",
            ".mov": "mov",
            ".avi": "avi",
        }
        return formatMap.get(ext, "mp4")

    def _buildFilterList(self):
        """Build list of video filters based on settings"""
        filterList = []

        if self.output_scale_width and self.output_scale_height:
            filterList.append(
                f"scale={self.output_scale_width}:{self.output_scale_height}:flags=bilinear"
            )

        if self.sharpen:
            filterList.append(f"cas={self.sharpen_sens}")
        if self.grayscale:
            filterList.append(
                "format=gray" if self.bitDepth == "8bit" else "format=gray16be"
            )
        if self.transparent:
            filterList.append("format=yuva420p")

        """
                "-vf",
            "zscale=matrix=709:dither=error_diffusion,format=yuv420p",
            """

        import json

        metadata = json.loads(open(cs.METADATAPATH, "r", encoding="utf-8").read())
        if not self.grayscale and not self.transparent:
            colorSPaceFilter = {
                "bt709": f"zscale=matrix=709:dither=error_diffusion,format={self.inputPixFmt}",
                "bt2020": "zscale=matrix=bt2020:norm=bt2020:dither=error_diffusion,format=yuv420p",
            }

            metadataFields = ["ColorSpace", "PixelFormat", "ColorTRT"]
            detectedColorSpace = None

            for field in metadataFields:
                colorValue = metadata["metadata"].get(field, "unknown")
                if colorValue in colorSPaceFilter:
                    detectedColorSpace = colorValue
                    break

            filterList.append(
                colorSPaceFilter.get(detectedColorSpace, colorSPaceFilter["bt709"])
            )

        return filterList

    def _buildCustomEncoder(self, filterList, outputPixFmt):
        """Apply custom encoder settings with filters"""
        customEncoderArgs = self.custom_encoder.split()

        if "-vf" in customEncoderArgs:
            vfIndex = customEncoderArgs.index("-vf")
            filterString = customEncoderArgs[vfIndex + 1]
            for filterItem in filterList:
                filterString += f",{filterItem}"
            customEncoderArgs[vfIndex + 1] = filterString
        elif filterList:
            customEncoderArgs.extend(["-vf", ",".join(filterList)])

        if "-pix_fmt" not in customEncoderArgs:
            logging.info(f"-pix_fmt was not found, adding {outputPixFmt}.")
            customEncoderArgs.extend(["-pix_fmt", outputPixFmt])

        return customEncoderArgs

    def _buildAudioSettings(self):
        """Build audio encoding settings"""
        audioSettings = ["-map", "1:a"]

        audioCodec = "copy"
        subCodec = "copy"
        if self.output.endswith(".webm"):
            audioCodec = "libopus"
            subCodec = "webvtt"
        audioSettings.extend(["-c:a", audioCodec, "-map", "1:s?", "-c:s", subCodec])

        if self.outpoint != 0:
            audioSettings.extend(["-ss", str(self.inpoint), "-to", str(self.outpoint)])

        return audioSettings

    def __call__(self):
        writtenFrames = 0

        # Wait for at least one frame to be queued before starting encoding
        while self.writeBuffer.empty():
            try:
                time.sleep(0.001)
            except KeyboardInterrupt:
                logging.warning("Encoding interrupted by user")
                return

        ffmpegProc = None
        try:
            import torch
            from torch.nn import functional as F

            initialFrame = self.writeBuffer.queue[0]

            self.channels = 1 if self.grayscale else 4 if self.transparent else 3

            isEightBit = self.bitDepth == "8bit"
            multiplier = 255 if isEightBit else 65535
            dtype = torch.uint8 if isEightBit else torch.uint16

            needsResize = (
                initialFrame.shape[2] != self.height
                or initialFrame.shape[3] != self.width
            )

            if needsResize:
                logging.info(
                    f"Frame size mismatch. Frame: {initialFrame.shape[3]}x{initialFrame.shape[2]}, Output: {self.width}x{self.height}"
                )

            command = self.encodeSettings()
            logging.info(f"Encoding with: {' '.join(map(str, command))}")

            if self.enablePreview:
                logging.info(f"Preview enabled, writing to: {self.previewPath}")
                from src.module.utils.logAndPrint import logAndPrint

                logAndPrint(f"Preview will be saved to: {self.previewPath}", "cyan")

            useCuda = False
            transferStream = None
            if checker.cuda_available:
                try:
                    transferStream = torch.cuda.Stream()
                    useCuda = True
                except Exception as e:
                    logging.warning(
                        f"CUDA init failed in writer, using CPU path. Reason: {e}"
                    )
                    useCuda = False

            ffmpegProc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=None,
                stderr=subprocess.DEVNULL,
                shell=False,
                cwd=cs.WHEREAMIRUNFROM,
            )

            if useCuda:
                frameShape = (self.height, self.width, self.channels)
                pinnedBuffers = [
                    torch.empty(frameShape, dtype=dtype, pin_memory=True),
                    torch.empty(frameShape, dtype=dtype, pin_memory=True),
                ]
                transferEvents = [torch.cuda.Event(), torch.cuda.Event()]
                bufferIdx = 0
                pendingBuffer = None
                pendingEvent = None

                while True:
                    try:
                        frame = self.writeBuffer.get(timeout=1.0)
                    except Exception:
                        time.sleep(0.001)
                        continue
                    if frame is None:
                        if pendingBuffer is not None:
                            pendingEvent.synchronize()
                            ffmpegProc.stdin.write(memoryview(pendingBuffer.numpy()))
                            writtenFrames += 1
                        break

                    with torch.cuda.stream(transferStream):
                        if needsResize:
                            frame = F.interpolate(
                                frame,
                                size=(self.height, self.width),
                                mode="bicubic",
                                align_corners=False,
                            )

                        gpuTensor = (
                            frame.squeeze(0)
                            .permute(1, 2, 0)
                            .mul(multiplier)
                            .clamp(0, multiplier)
                            .to(dtype)
                            .contiguous()
                        )

                        currentBuffer = pinnedBuffers[bufferIdx]
                        currentBuffer.copy_(gpuTensor, non_blocking=True)
                        currentEvent = transferEvents[bufferIdx]
                        currentEvent.record(transferStream)

                    if pendingBuffer is not None:
                        pendingEvent.synchronize()
                        ffmpegProc.stdin.write(memoryview(pendingBuffer.numpy()))
                        writtenFrames += 1

                    pendingBuffer = currentBuffer
                    pendingEvent = currentEvent
                    bufferIdx = 1 - bufferIdx

            else:
                while True:
                    try:
                        frame = self.writeBuffer.get(timeout=1.0)
                    except Exception:
                        time.sleep(0.001)
                        continue
                    if frame is None:
                        break

                    if needsResize:
                        frame = F.interpolate(
                            frame,
                            size=(self.height, self.width),
                            mode="bicubic",
                            align_corners=False,
                        )
                    frameTensor = (
                        frame.squeeze(0)
                        .permute(1, 2, 0)
                        .mul(multiplier)
                        .clamp(0, multiplier)
                        .to(dtype)
                        .contiguous()
                    )

                    ffmpegProc.stdin.write(memoryview(frameTensor.numpy()))
                    writtenFrames += 1

            logging.info(f"Encoded {writtenFrames} frames")

        except Exception as e:
            logging.error(f"Encoding error: {e}")
        finally:
            try:
                if ffmpegProc is not None and ffmpegProc.stdin:
                    ffmpegProc.stdin.close()
                if ffmpegProc is not None:
                    ffmpegProc.wait(timeout=3)

            except Exception as e:
                logging.warning(f"Cleanup error: {e}")

    def write(self, frame):
        """
        Add a frame to the queue. Must be in [B, C, H, W] format (BCHW).
        Frame type is torch.Tensor when using PyTorch backend.
        """
        self.writeBuffer.put(frame)

    def put(self, frame):
        """
        Equivalent to write()
        Add a frame to the queue. Must be in [B, C, H, W] format (BCHW).
        Frame type is torch.Tensor when using PyTorch backend.
        """
        self.writeBuffer.put(frame)

    def close(self):
        self.writeBuffer.put(None)

        if self.previewPath and os.path.exists(self.previewPath):
            try:
                os.remove(self.previewPath)
            except Exception as e:
                logging.warning(f"Could not remove preview file: {e}")


class NeluxWriteBuffer:
    """
    Write buffer that uses Nelux VideoEncoder for NVENC encoding.
    More efficient than FFmpeg pipe for GPU-resident frames.
    """

    def __init__(
        self,
        input: str = "",
        output: str = "",
        encode_method: str = "h264_nvenc_nelux",
        width: int = 1920,
        height: int = 1080,
        fps: float = 60.0,
        inpoint: float = 0.0,
        outpoint: float = 0.0,
        **kwargs,  # Accept and ignore other WriteBuffer params for compatibility
    ):
        """
        Initialize Nelux-based encoder.

        Args:
            input: Input video path (for audio extraction).
            output: Output video path.
            encode_method: One of nvenc_h264_nelux, nvenc_h265_nelux, nvenc_av1_nelux.
            width: Output width.
            height: Output height.
            fps: Output framerate.
            inpoint: Start time for audio (seconds).
            outpoint: End time for audio (seconds).
        """
        self.input = input
        self.output = os.path.normpath(output)
        self.width = width
        self.height = height
        self.fps = fps
        self.inpoint = inpoint
        self.outpoint = outpoint
        self.writeBuffer = Queue(maxsize=64)
        self.writtenFrames = 0
        self.CudaStream = None

        codec_map = {
            "nvenc_h264_nelux": "h264_nvenc",
            "nvenc_h265_nelux": "hevc_nvenc",
            "nvenc_av1_nelux": "av1_nvenc",
        }
        self.codec = codec_map.get(encode_method, "h264_nvenc")
        self.encoder = None

        if checker.cuda_available:
            self.CudaStream = torch.cuda.Stream()

        logging.info(
            f"NeluxWriteBuffer initialized: {width}x{height}@{fps}fps, codec={self.codec}"
        )

    def __call__(self):
        """Process frames from writeBuffer and encode with Nelux."""
        import torch

        try:
            while self.writeBuffer.empty():
                time.sleep(0.001)

            self.encoder = nelux.VideoEncoder(
                self.output,
                codec=self.codec,
                width=self.width,
                height=self.height,
                fps=self.fps,
            )

            if hasattr(self.encoder, "is_hardware_encoder"):
                if self.encoder.is_hardware_encoder:
                    logging.info(
                        f"Nelux NVENC encoder confirmed: {self.codec} -> {self.output}"
                    )
                else:
                    logging.warning(
                        f"Nelux encoder is NOT using hardware NVENC! Codec: {self.codec}"
                    )
            else:
                logging.info(f"Nelux encoder created: {self.codec} -> {self.output}")

            while True:
                try:
                    frame = self.writeBuffer.get(timeout=1.0)
                except Exception:
                    time.sleep(0.001)
                    continue

                if frame is None:
                    break

                with torch.cuda.stream(self.CudaStream):
                    frame = frame.squeeze(0).permute(1, 2, 0)
                    frame = (
                        frame.mul(255.0)
                        .clamp(0, 255)
                        .to(dtype=torch.uint8, non_blocking=True)
                    )
                self.CudaStream.synchronize()

                if not frame.is_contiguous():
                    frame = frame.contiguous()

                self.encoder.encode_frame(frame)
                self.writtenFrames += 1

            logging.info(f"Nelux encoded {self.writtenFrames} frames")

        except Exception as e:
            logging.error(f"Nelux encoding error: {e}")
            import traceback

            traceback.print_exc()
        finally:
            if self.encoder is not None:
                try:
                    self.encoder.close()
                except Exception as e:
                    logging.warning(f"Error closing Nelux encoder: {e}")

    def write(self, frame):
        """Add a frame to the queue. Must be in [H, W, 3] HWC format."""
        self.writeBuffer.put(frame)

    def put(self, frame):
        """Equivalent to write(). Add a frame to the queue."""
        self.writeBuffer.put(frame)

    def close(self):
        """Signal end of encoding."""
        self.writeBuffer.put(None)


def isNeluxEncoder(encode_method: str) -> bool:
    """Check if the encode method uses Nelux NVENC."""
    return encode_method.endswith("_nelux")


def createWriteBuffer(encode_method: str, **kwargs):
    """
    Factory function to create the appropriate write buffer.

    Args:
        encode_method: The encoding method string.
        **kwargs: Arguments passed to the buffer constructor.

    Returns:
        WriteBuffer or NeluxWriteBuffer instance.

    Usage:
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
    if isNeluxEncoder(encode_method):
        logging.info(f"Using NeluxWriteBuffer for {encode_method}")
        return NeluxWriteBuffer(encode_method=encode_method, **kwargs)
    else:
        return WriteBuffer(encode_method=encode_method, **kwargs)
