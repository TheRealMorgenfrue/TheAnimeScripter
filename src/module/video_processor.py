import os
import queue
import traceback
from concurrent.futures import ThreadPoolExecutor
from math import ceil
from time import time

from applib import LoggingManager
from torch import Tensor
from torch.profiler import ProfilerActivity, profile
from tqdm import tqdm

from src.module.config.tas_args import TASArgs
from src.module.config.tas_config import TASConfig
from src.module.io.get_video_metadata import get_video_metadata
from src.module.io.io_buffers import (
    PeekQueue,
    ReadBuffer,
    create_write_buffer,
)
from src.module.io.io_handler import PathConfiguration
from src.module.models.model_base import ModelBase
from src.module.models.model_handler import ModelHandler
from src.module.utils.cuda_checker import CudaChecker


class VideoProcessor:
    """
    Main video processing class that handles AI-powered video enhancement operations.

    Supports upscaling, interpolation, restoration, deduplication, and various other
    video processing operations using different AI models and hardware backends.
    """

    def __init__(self, path_config: PathConfiguration):
        self.start_time: float = time()
        self.logger = LoggingManager()
        self.config = TASConfig()
        self.error_buffer = []  # Stores errors from processing threads

        # Input args
        self.path = path_config
        self.input_metadata = get_video_metadata(self.path.input_path)
        self.width = self.input_metadata["width"]
        self.height = self.input_metadata["height"]
        self.fps = self.input_metadata["fps"]

        # Config args
        self.vfi: bool = self.config["vfi"]
        self.vfi_factor: float = self.config["vfi_factor"]
        self.vfi_model: str = self.config["vfi_model"]
        self.sr: bool = self.config["sr"]

        self._configure_processing_options()  # Must be called before "Frame processing"

        # Frame processing
        self.should_get_next_frame = self.config["sr_model"] == "animesr" or (
            self.vfi and self.vfi_model.startswith(("distildrba", "atr"))
        )
        self.process_list: list[ModelBase] = ModelHandler().initialize_models(
            width=self.width, height=self.height
        )
        self.current_frame_buffer: PeekQueue[Tensor] = PeekQueue(
            maxsize=ceil(self.vfi_factor)
        )
        self.read_buffer = ReadBuffer(
            input_path=self.path.input_path,
            width=self.width,
            height=self.height,
        )
        self.write_buffer = create_write_buffer(
            encode_method=self.config["encode_method"],
            input_path=self.path.input_path,
            output_path=self.path.output_path,
            width=self.width,
            height=self.height,
            fps=self.fps,
            input_metadata_config=self.input_metadata,
            grayscale=False,
            transparent=False,
        )

        self._execute_pipeline()

    def _configure_processing_options(self) -> None:
        """Configure processing options based on the selected operations."""
        if self.vfi:
            self.fps *= self.vfi_factor
        if self.sr:
            sr_factor = self.config["sr_factor"]
            self.width *= sr_factor
            self.height *= sr_factor

    def _execute_pipeline(self) -> None:
        """Select and execute the appropriate processing method based on user options."""
        # TODO: Add scene detection

        self.start()

    def _process_frame(self, frame: Tensor) -> None:
        """
        Process a single video frame through the configured enhancement pipeline.
        """
        self.current_frame_buffer.put_nowait(frame)
        next_frame = self.read_buffer.peek() if self.should_get_next_frame else None
        for process in self.process_list:
            for _ in range(len(self.current_frame_buffer.queue)):
                try:
                    current_frame = self.current_frame_buffer.get_nowait()
                except queue.Empty:
                    self.logger.critical(
                        "Current frame buffer is unexpectedly empty. This is very bad",
                        gui=True,
                        pid=0,
                    )
                    break

                if self.should_get_next_frame:
                    try:
                        current_next_frame = self.current_frame_buffer.peek()
                    except queue.Empty:
                        current_next_frame = next_frame
                else:
                    current_next_frame = next_frame

                frame_predictions = process.inference(
                    frame=current_frame,
                    next_frame=current_next_frame,
                )
                for frame in frame_predictions:
                    self.current_frame_buffer.put_nowait(frame)

        for _ in range(len(self.current_frame_buffer.queue)):
            self.write_buffer.write(self.current_frame_buffer.get_nowait())

    def _process(self):
        """
        Main processing loop that handles frame-by-frame video processing.

        Processes all frames through the configured enhancement pipeline and
        tracks processing statistics.
        """
        increment = self.vfi_factor if self.vfi else 1
        total_frames_to_process = self.input_metadata["total_frames_to_process"]

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
                        f"Frame buffer ended unexpectedly. The output is most likely incomplete. Processed {i} of {total_frames_to_process} frames",
                        gui=True,
                        pid=0,
                    )
                    break

                self._process_frame(frame)
            self.write_buffer.close()
        except Exception:
            self.logger.error(
                f"Failed to process frame:\n{traceback.format_exc()}", gui=True, pid=0
            )

    def start(self):
        """
        Initialize and start the video processing pipeline.

        Sets up input/output buffers, initializes AI models, and coordinates
        the multi-threaded processing workflow.
        """

        if self.config["profile"]:
            self._run_with_profiler()
        else:
            with ThreadPoolExecutor(max_workers=3) as executor:
                executor.submit(self.read_buffer)
                executor.submit(self.write_buffer)
                executor.submit(self._process)

        total_frames_to_process = self.input_metadata["total_frames_to_process"]
        elapsed_time: float = time() - self.start_time
        total_fps: float = (
            total_frames_to_process
            / elapsed_time
            * (1 if not self.vfi else self.vfi_factor)
        )

        self.logger.info(
            f"Total Execution Time: {elapsed_time:.2f} seconds - FPS: {total_fps:.2f}",
            pid=0,
        )

    def _run_with_profiler(self):
        """
        Run the processing pipeline with torch.profiler enabled.
        Uses a simplified approach compatible with multi-threaded execution on Windows.
        """
        is_cuda_available = CudaChecker().cuda_available
        profilePath = os.path.join(TASArgs.app_dir, "profiler_trace")
        os.makedirs(profilePath, exist_ok=True)

        self.logger.info(
            f"Profiling enabled. Trace will be saved to: {profilePath}", pid=0
        )

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

        self.logger.info("=== Profiler Summary (Top 20 by CUDA time) ===", pid=0)

        try:
            sortKey = "cuda_time_total" if is_cuda_available else "cpu_time_total"
            summary = prof.key_averages().table(sort_by=sortKey, row_limit=20)
            self.logger.info(f"Profiler Summary:\n\t{summary}", pid=0)
        except Exception:
            self.logger.error(
                f"Could not print profiler summary\n{traceback.format_exc()}", pid=0
            )
        self.logger.info(f"Trace saved to: {traceFile}", pid=0)
