import logging
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from queue import Queue
from time import time

import torch
from applib import LoggingManager
from torch import Tensor
from torch.profiler import ProfilerActivity, profile
from tqdm import tqdm

from module.config.tas_args import TASArgs
from module.config.tas_config import TASConfig
from module.initializeModels import initializeModels
from module.utils.getVideoMetadata import get_video_metadata
from module.utils.io_buffers import ReadBuffer, createWriteBuffer
from module.utils.progressBarLogic import ProgressBarLogic
from src.module.utils.inputOutputHandler import PathConfiguration


class VideoProcessor:
    """
    Main video processing class that handles AI-powered video enhancement operations.

    Supports upscaling, interpolation, restoration, deduplication, and various other
    video processing operations using different AI models and hardware backends.
    """

    def __init__(self, path_config: PathConfiguration):
        self.logger = LoggingManager()
        self.config = TASConfig()
        self.error_buffer = []  # Stores processing thread errors

        self.input = path_config
        self.input_metadata = get_video_metadata(self.input.input_path)

        self.output_fps = self.input_metadata.get_value("fps")

        # Frame processing
        self.dedup_count = 0
        self.frame_counter = 0
        self.next_frame = None
        self.timesteps = None
        self.frames_to_insert = 0

        self._configure_processing_options()
        self._execute_pipeline()

    def _configure_processing_options(self) -> None:
        """Configure processing options based on the selected operations."""
        if self.config.get_value("vfi"):
            self.output_fps = self.output_fps * self.config.get_value("vfi_factor")

        if self.config.get_value("resize"):
            resize_factor = self.config.get_value("resize_factor")
            aspect_ratio = self.input_metadata.get_value("aspect_ratio")
            old_width = self.input_metadata.get_value("width")
            new_width = round(old_width * resize_factor / 2) * 2
            new_height = round(old_width / aspect_ratio / 2) * 2
            self.logger.info(
                f"Resizing to {new_width}x{new_height} using {resize_factor} factor."
            )
            self.input_metadata.set_value("width", new_width, "Video")
            self.input_metadata.set_value("height", new_height, "Video")

    def _execute_pipeline(self) -> None:
        """
        Select and execute the appropriate processing method based on user options.

        Prioritizes specialized operations (autoclip, depth, segment, object detection)
        over standard video processing.
        """
        if self.config.get_value("depth"):
            self.logger.info("Depth Estimation")

            from src.module.initializeModels import depth

            depth(self)
        elif self.config.get_value("segment"):
            self.logger.info("Segmenting video")

            from src.module.initializeModels import segment

            segment(self)
        elif self.config.get_value("obj_detect"):
            self.logger.info("Object Detection")

            from src.module.initializeModels import objectDetection

            objectDetection(self)
        else:
            self.start()

    def process_frame(self, frame: Tensor) -> None:
        """
        Process a single video frame through the configured enhancement pipeline.
        """
        if self.dedup and self.dedup_process(frame):
            self.dedup_count += 1
            return

        if self.restore:
            frame = self.restore_process(frame)

        if self.interpolate:
            if isinstance(self.interpolateFactor, float):
                currentIDX = self.frame_counter
                nextIDX = currentIDX + 1

                outputStart = (currentIDX * self.factorNum) // self.factorDen
                outputEnd = (nextIDX * self.factorNum) // self.factorDen

                self.frames_to_insert = outputEnd - outputStart - 1

                self.timesteps = []
                for i in range(1, self.frames_to_insert + 1):
                    outputIDX = outputStart + i
                    t = (outputIDX * self.factorDen % self.factorNum) / self.factorNum
                    self.timesteps.append(t)

                self.frame_counter += 1
            else:
                self.frames_to_insert = int(self.interpolateFactor) - 1
                self.timesteps = None

        if self.interpolateFirst:
            self.ifInterpolateFirst(frame)
        else:
            self.ifInterpolateLast(frame)

    def ifInterpolateFirst(self, frame: Tensor) -> None:
        """Process frame with interpolation-first pipeline order."""
        if self.interpolate:
            if self.interpolateMethod.startswith(("distildrba", "atr")):
                self.interpolate_process(
                    frame,
                    self.next_frame,
                    self.interpQueue,
                    self.frames_to_insert,
                    self.timesteps,
                )
            else:
                self.interpolate_process(
                    frame, self.interpQueue, self.frames_to_insert, self.timesteps
                )

        if self.upscale:
            if self.interpolate:
                while not self.interpQueue.empty():
                    self.writeBuffer.write(
                        self.upscale_process(self.interpQueue.get(), self.next_frame)
                    )

                self.writeBuffer.write(self.upscale_process(frame, self.next_frame))

            else:
                self.writeBuffer.write(self.upscale_process(frame, self.next_frame))

        else:
            if self.interpolate:
                while not self.interpQueue.empty():
                    self.writeBuffer.write(self.interpQueue.get())
            self.writeBuffer.write(frame)

    def ifInterpolateLast(self, frame: Tensor) -> None:
        """Process frame with interpolation-last pipeline order."""
        if self.upscale:
            frame = self.upscale_process(frame, self.next_frame)

        if self.interpolate:
            if self.interpolateMethod.startswith(("distildrba", "atr")):
                self.interpolate_process(
                    frame,
                    self.next_frame,
                    self.writeBuffer,
                    self.frames_to_insert,
                    self.timesteps,
                )
            else:
                self.interpolate_process(
                    frame, self.writeBuffer, self.frames_to_insert, self.timesteps
                )

        self.writeBuffer.write(frame)

    def process(self):
        """
        Main processing loop that handles frame-by-frame video processing.

        Processes all frames through the configured enhancement pipeline and
        tracks processing statistics.
        """
        frameCount = 0

        if self.interpolate and isinstance(self.interpolateFactor, float):
            factor = Fraction(self.interpolateFactor).limit_denominator(100)
            self.factorNum = factor.numerator
            self.factorDen = factor.denominator

            increment = self.factorNum / self.factorDen
            if increment.is_integer():
                increment = int(increment)
        else:
            self.factorNum = self.interpolateFactor if self.interpolate else 1
            self.factorDen = 1
            increment = int(self.interpolateFactor) if self.interpolate else 1

        self.timesteps = None
        self.frames_to_insert = self.interpolateFactor - 1 if self.interpolate else 0

        if self.interpolate and self.interpolateFirst:
            self.interpQueue = Queue(maxsize=round(self.interpolateFactor))

        try:
            for _ in tqdm(
                range(self.total_frames),
                miniters=0,
                ascii=False,
                unit="FPS",
                dynamic_ncols=True,
                smoothing=0.5,
                postfix={},
                colour="#00ff00",
            ):
                frame = self.readBuffer.read()

                if frame is None:
                    # End of framebuffer
                    break

                if self.upscaleMethod == "animesr" or (
                    self.interpolate
                    and self.interpolateMethod.startswith(("distildrba", "atr"))
                ):
                    self.next_frame = self.readBuffer.peek()
                self.process_frame(frame)

            self.writeBuffer.close()

        except Exception as e:
            self.error_buffer.append(e)

        self.logger.info(f"Processed {frameCount} frames")
        if self.dedup_count > 0:
            self.logger.info(f"Deduplicated {self.dedup_count} frames")

    def start(self):
        """
        Initialize and start the video processing pipeline.

        Sets up input/output buffers, initializes AI models, and coordinates
        the multi-threaded processing workflow.
        """
        (
            self.new_width,
            self.new_height,
            self.upscale_process,
            self.interpolate_process,
            self.restore_process,
            self.dedup_process,
        ) = initializeModels(self)

        start_time: float = time()

        self.readBuffer = ReadBuffer(
            video_input=self.input,
            inpoint=self.inpoint,
            outpoint=self.outpoint,
            half=self.half,
            resize=self.resize,
            width=self.width,
            height=self.height,
            bit_depth=self.bitDepth,
            decode_method=self.decodeMethod,
        )

        self.writeBuffer = createWriteBuffer(
            input=self.input,
            output=self.output,
            encode_method=self.encodeMethod,
            custom_encoder=self.customEncoder,
            width=self.new_width,
            height=self.new_height,
            fps=self.outputFPS,
            sharpen=self.sharpen,
            sharpen_sens=self.sharpenSens,
            grayscale=False,
            transparent=False,
            benchmark=self.benchmark,
            bitDepth=self.bitDepth,
            inpoint=self.inpoint,
            outpoint=self.outpoint,
            slowmo=self.slowmo,
            output_scale_width=self.outputScaleWidth,
            output_scale_height=self.outputScaleHeight,
        )

        if self.config.get_value("profile"):
            self._run_with_profiler()
        else:
            with ThreadPoolExecutor(max_workers=3) as executor:
                executor.submit(self.readBuffer)
                executor.submit(self.writeBuffer)
                executor.submit(self.process)

        elapsed_time: float = time() - start_time
        total_fps: float = (
            self.total_frames
            / elapsed_time
            * (1 if not self.interpolate else self.interpolateFactor)
        )

        self.logger.info(
            f"Total Execution Time: {elapsed_time:.2f} seconds - FPS: {total_fps:.2f}",
        )

    def _run_with_profiler(self):
        """
        Run the processing pipeline with torch.profiler enabled.
        Uses a simplified approach compatible with multi-threaded execution on Windows.
        """
        profilePath = os.path.join(TASArgs.app_dir, "profiler_trace")
        os.makedirs(profilePath, exist_ok=True)

        self.logger.info(f"Profiling enabled. Trace will be saved to: {profilePath}")

        activities = [ProfilerActivity.CPU]

        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            with ThreadPoolExecutor(max_workers=3) as executor:
                executor.submit(self.readBuffer)
                executor.submit(self.writeBuffer)
                executor.submit(self.process)

        traceFile = os.path.join(profilePath, "trace.json")
        prof.export_chrome_trace(traceFile)

        self.logger.info("=== Profiler Summary (Top 20 by CUDA time) ===")

        try:
            sortKey = (
                "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
            )
            summary = prof.key_averages().table(sort_by=sortKey, row_limit=20)
            self.logger.info(f"Profiler Summary:\n\t{summary}")
        except Exception:
            self.logger.error(
                f"Could not print profiler summary\n{traceback.format_exc()}"
            )

        self.logger.info(f"Trace saved to: {traceFile}")
