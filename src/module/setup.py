import traceback
from time import time

from applib import LoggingManager

from src.module.io.io_handler import IOHandler
from src.module.video_processor import VideoProcessor


def process_arguments():
    """Execute processing pipeline based on config values."""
    start_time = time()
    logger = LoggingManager()
    input_files = IOHandler().get_input_files()
    file_count = len(input_files)

    for i, path_config in enumerate(input_files, 1):
        logger.info(f"({i}/{file_count}) Processing video '{path_config.input_path}'")
        try:
            VideoProcessor(path_config)
        except Exception:
            logger.error(
                f"Failed to process video '{path_config.input_path}'\n{traceback.format_exc()}"
            )

    logger.info(f"Total execution time: {time() - start_time:.2f} s")
