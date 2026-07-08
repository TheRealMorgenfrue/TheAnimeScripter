import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from math import ceil
from queue import Queue
from time import time

from applib import LoggingManager
from torch import Tensor
from torch.profiler import ProfilerActivity, profile
from tqdm import tqdm

from module.io.get_video_metadata import get_video_metadata
from module.io.io_buffers import (
    ReadBuffer,
    WriteBuffer,
    create_write_buffer,
)
from module.io.io_handler import PathConfiguration
from src.module.config.tas_args import TASArgs
from src.module.config.tas_config import TASConfig
from src.module.initializeModels import initialize_models
from src.module.utils.cuda_checker import CudaChecker


class VideoProcessor:
    """
    Main video processing class that handles AI-powered video enhancement operations.

    Supports upscaling, interpolation, restoration, deduplication, and various other
    video processing operations using different AI models and hardware backends.
    """

    def __init__(self, path_config: PathConfiguration):
        self.logger = LoggingManager()
        self.config = TASConfig()
        self.error_buffer = []  # Stores errors from processing threads

        self.path = path_config
        self.input_metadata = get_video_metadata(self.path.input_path)

        # Frame processing
        self.dedup_count = 0
        self.frame_counter = 0
        self.timesteps: list[float] | None = None
        self.vfi_factor_numerator = 1
        self.vfi_factor_denominator = 1
        self.vfi_buffer: Queue | None = None
        self.read_buffer: ReadBuffer | None = None
        self.write_buffer: WriteBuffer | None = None

        # Config args
        self.dedup: bool = self.config.get_value("dedup")
        self.vfi: bool = self.config.get_value("vfi")
        self.vfi_factor: float = self.config.get_value("vfi_factor")
        self.vfi_model: str = self.config.get_value("vfi_model")
        self.vfi_first: bool = self.config.get_value("vfi_first")
        self.sr: bool = self.config.get_value("sr")

        self._configure_processing_options()
        self._execute_pipeline()

    def _configure_processing_options(self) -> None:
        """Configure processing options based on the selected operations."""
        if self.vfi:
            new_fps = self.input_metadata.get_value("fps") * self.vfi_factor
            self.input_metadata.set_value("fps", new_fps)

        if self.sr:
            sr_factor = self.config.get_value("sr_factor")
            new_width = self.input_metadata.get_value("width") * sr_factor
            new_height = self.input_metadata.get_value("height") * sr_factor
            self.input_metadata.set_value("width", new_width)
            self.input_metadata.set_value("height", new_height)

    def _execute_pipeline(self) -> None:
        """Select and execute the appropriate processing method based on user options."""
        # TODO: Add scene detection (called Autoclip in TAS)

        self.start()

    def _process_frame(self, frame: Tensor, next_frame: Tensor | None) -> None:
        """
        Process a single video frame through the configured enhancement pipeline.
        """

        if self.dedup and self.dedup_process(frame):
            self.dedup_count += 1
            return

        if self.vfi:
            if isinstance(self.vfi, float):
                current_index = self.frame_counter
                next_index = current_index + 1

                output_start = (
                    current_index * self.vfi_factor_numerator
                ) // self.vfi_factor_denominator
                output_end = (
                    next_index * self.vfi_factor_numerator
                ) // self.vfi_factor_denominator

                frames_to_insert = output_end - output_start - 1

                self.timesteps = []
                for i in range(1, frames_to_insert + 1):
                    outputIDX = output_start + i
                    t = (
                        outputIDX
                        * self.vfi_factor_denominator
                        % self.vfi_factor_numerator
                    ) / self.vfi_factor_numerator
                    self.timesteps.append(t)

                self.frame_counter += 1
            else:
                frames_to_insert = int(self.vfi_factor) - 1
                self.timesteps = None

        if self.vfi_first:
            self._vfi_first(frame, next_frame, frames_to_insert)
        else:
            self._vfi_last(frame, next_frame, frames_to_insert)

    def _vfi_first(
        self, frame: Tensor, next_frame: Tensor | None, frames_to_insert: int
    ) -> None:
        """Process frame with interpolation-first pipeline order."""
        if self.vfi:
            if next_frame is not None:
                self.interpolate_process(
                    frame,
                    next_frame,
                    self.vfi_buffer,
                    frames_to_insert,
                    self.timesteps,
                )
            else:
                self.interpolate_process(
                    frame, self.vfi_buffer, frames_to_insert, self.timesteps
                )

        if self.sr:
            if self.vfi:
                while not self.vfi_buffer.empty():
                    self.write_buffer.write(
                        self.upscale_process(self.vfi_buffer.get(), next_frame)
                    )
            self.write_buffer.write(self.upscale_process(frame, next_frame))

        else:
            if self.vfi:
                while not self.vfi_buffer.empty():
                    self.write_buffer.write(self.vfi_buffer.get())
            self.write_buffer.write(frame)

    def _vfi_last(
        self, frame: Tensor, next_frame: Tensor | None, frames_to_insert: int
    ) -> None:
        """Process frame with interpolation-last pipeline order."""
        if self.sr:
            frame = self.upscale_process(frame, next_frame)

        if self.vfi:
            if next_frame is not None:
                self.interpolate_process(
                    frame,
                    next_frame,
                    self.write_buffer,
                    frames_to_insert,
                    self.timesteps,
                )
            else:
                self.interpolate_process(
                    frame, self.write_buffer, frames_to_insert, self.timesteps
                )

        self.write_buffer.write(frame)

    def _process(self):
        """
        Main processing loop that handles frame-by-frame video processing.

        Processes all frames through the configured enhancement pipeline and
        tracks processing statistics.
        """
        increment = 1
        total_frames_to_process = self.input_metadata.get_value(
            "total_frames_to_process"
        )
        should_get_next_frame = self.config.get_value("sr_model") == "animesr" or (
            self.vfi and self.vfi_model.startswith(("distildrba", "atr"))
        )

        if self.vfi:
            increment = self.vfi_factor

            if isinstance(self.vfi_factor, float):
                factor = Fraction(self.vfi_factor).limit_denominator(100)
                self.vfi_factor_numerator = factor.numerator
                self.vfi_factor_denominator = factor.denominator
            else:
                self.vfi_factor_numerator = self.vfi_factor

            if self.vfi_first:
                self.vfi_buffer = Queue(maxsize=ceil(self.vfi_factor))

        try:
            for i in tqdm(
                range(total_frames_to_process),
                total=total_frames_to_process,  # * increment,
                miniters=0,
                ascii=False,
                unit="FPS",
                dynamic_ncols=True,
                smoothing=0.5,
                postfix={},
                colour="#00ff00",
            ):
                frame = self.read_buffer.read()

                if frame is None:
                    # End of framebuffer
                    self.logger.warning(
                        f"Frame buffer ended unexpectedly - the output is most likely incomplete. Processed {i} of {total_frames_to_process} frames"
                    )
                    break

                next_frame = self.read_buffer.peek() if should_get_next_frame else None
                self._process_frame(frame, next_frame)
            self.write_buffer.close()
        except Exception as e:
            self.error_buffer.append(e)

    def start(self):
        """
        Initialize and start the video processing pipeline.

        Sets up input/output buffers, initializes AI models, and coordinates
        the multi-threaded processing workflow.
        """
        (
            self.upscale_process,
            self.interpolate_process,
            self.restore_process,
            self.dedup_process,
        ) = initialize_models(self)

        start_time: float = time()

        width = self.input_metadata.get_value("width")
        height = self.input_metadata.get_value("height")

        self.read_buffer = ReadBuffer(
            input_path=self.path.input_path,
            width=width,
            height=height,
        )

        self.write_buffer = create_write_buffer(
            encode_method=self.config.get_value("encode_method"),
            input=self.path.input_path,
            output=self.path.output_path,
            width=width,
            height=height,
            fps=self.input_metadata.get_value("fps"),
            grayscale=False,
            transparent=False,
        )

        if self.config.get_value("profile"):
            self._run_with_profiler()
        else:
            with ThreadPoolExecutor(max_workers=3) as executor:
                executor.submit(self.read_buffer)
                executor.submit(self.write_buffer)
                executor.submit(self._process)

        total_frames_to_process = self.input_metadata.get_value(
            "total_frames_to_process"
        )
        elapsed_time: float = time() - start_time
        total_fps: float = (
            total_frames_to_process
            / elapsed_time
            * (1 if not self.vfi else self.vfi_factor)
        )

        self.logger.info(
            f"Total Execution Time: {elapsed_time:.2f} seconds - FPS: {total_fps:.2f}",
        )

    def _run_with_profiler(self):
        """
        Run the processing pipeline with torch.profiler enabled.
        Uses a simplified approach compatible with multi-threaded execution on Windows.
        """
        is_cuda_available = CudaChecker().cuda_available
        profilePath = os.path.join(TASArgs.app_dir, "profiler_trace")
        os.makedirs(profilePath, exist_ok=True)

        self.logger.info(f"Profiling enabled. Trace will be saved to: {profilePath}")

        activities = [ProfilerActivity.CPU]

        if is_cuda_available:
            activities.append(ProfilerActivity.CUDA)

        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            with ThreadPoolExecutor(max_workers=3) as executor:
                executor.submit(self.read_buffer)
                executor.submit(self.write_buffer)
                executor.submit(self._process)

        traceFile = os.path.join(profilePath, "trace.json")
        prof.export_chrome_trace(traceFile)

        self.logger.info("=== Profiler Summary (Top 20 by CUDA time) ===")

        try:
            sortKey = "cuda_time_total" if is_cuda_available else "cpu_time_total"
            summary = prof.key_averages().table(sort_by=sortKey, row_limit=20)
            self.logger.info(f"Profiler Summary:\n\t{summary}")
        except Exception:
            self.logger.error(
                f"Could not print profiler summary\n{traceback.format_exc()}"
            )
        self.logger.info(f"Trace saved to: {traceFile}")
