import glob
import os
import random
import re

from applib import LoggingManager

from src.module.config.tas_args import TASArgs
from src.module.config.tas_config import TASConfig


class PathConfiguration:
    def __init__(
        self,
        input_path: str,
        output_path: str,
    ) -> None:
        self.input_path = input_path
        self.output_path = output_path


class IOHandler:
    def __init__(self) -> None:
        self.logger = LoggingManager()

    def _detect_image_sequence(self, folder_path: str):
        """
        Detects if a folder contains an image sequence and returns the sequence pattern.

        Args:
            folderPath: Path to the folder to check

        Returns:
            tuple: (sequencePattern, firstFrame, lastFrame, frameCount) or None if not a sequence
        """
        if not os.path.isdir(folder_path):
            return None

        image_files = []
        for ext in TASArgs.image_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
            image_files.extend(glob.glob(os.path.join(folder_path, f"*{ext.upper()}")))

        if len(image_files) < 2:
            return None

        image_files.sort()

        first_file = os.path.basename(image_files[0])

        patterns = [
            r"^(.+?)(\d+)(\.[^.]+)$",
        ]

        for pattern in patterns:
            match = re.match(pattern, first_file)
            if match:
                prefix, number, extension = match.groups()
                padding = len(number)

                expected_pattern = f"{prefix}%0{padding}d{extension}"

                frame_numbers = []
                for img_file in image_files:
                    basename = os.path.basename(img_file)
                    m = re.match(pattern, basename)
                    if (
                        m
                        and m.group(1) == prefix
                        and m.group(3).lower() == extension.lower()
                    ):
                        try:
                            frame_numbers.append(int(m.group(2)))
                        except ValueError:
                            continue

                if len(frame_numbers) >= 2:
                    frame_numbers.sort()
                    first_frame = frame_numbers[0]
                    last_frame = frame_numbers[-1]

                    sequence_path = os.path.join(folder_path, expected_pattern)

                    return (sequence_path, first_frame, last_frame, len(frame_numbers))

        return None

    def _generate_output_name(self, config: TASConfig, input_path: str):
        """Generates output filename based on input and processing arguments."""

        ## Suffix arguments ##
        ensemble = "-ensemble" if config.get_value("ensemble", default=False) else ""
        dynamic_scale = (
            "-dynamic_scale" if config.get_value("dynamic_scale", default=False) else ""
        )

        argMap = {
            "resize": (
                f"_Resize{config.get_value('resize_factor', default='')}"
                if config.get_value("resize", default=False)
                else ""
            ),
            "dedup": (
                f"_Dedup={config.get_value('dedup_method', default='')}-Sens={config.get_value('dedup_sens', default='')}"
                if config.get_value("dedup", default=False)
                else ""
            ),
            "interpolate": (
                f"_VFI={config.get_value('interpolate_method', default='')}-{config.get_value('interpolate_factor', default='')}x{ensemble}{dynamic_scale}"
                if config.get_value("interpolate", default=False)
                else ""
            ),
            "upscale": (
                f"_SR={config.get_value('upscale_method', default='')}-{config.get_value('upscale_factor', default='')}x"
                if config.get_value("upscale", default=False)
                else ""
            ),
            "sharpen": (
                f"_Sh{config.get_value('sharpen_sens', default='')}"
                if config.get_value("sharpen", default=False)
                else ""
            ),
            "restore": (
                f"_Restore{config.get_value('restore_method', default='')}"
                if config.get_value("restore", default=False)
                else ""
            ),
            "segment": "_Segment" if config.get_value("segment", default=False) else "",
            "depth": "_Depth" if config.get_value("depth", default=False) else "",
            "ytdlp": "_YTDLP" if config.get_value("ytdlp", default=False) else "",
        }
        #####

        # Handle URL input
        if input_path in ["https://", "http://"]:
            return f"TAS-YTDLP-{random.randint(0, 1000)}.mp4"

        # Start with base name
        baseName = (
            os.path.splitext(os.path.basename(input_path))[0] if input_path else "TAS"
        )

        # Add processing indicators
        suffixes = [suffix for suffix in argMap.values() if suffix]

        # Add random number to prevent overwrites
        suffixes.append(f"_ID{random.randint(0, 1000)}")

        # Determine extension
        if (
            config.get_value("segment", default=False)
            or config.get_value("encode_method", "") == "prores"
        ):
            extension = ".mov"
        elif config.get_value("encode_method") == "png":
            extension = ""
        elif input_path:
            extension = os.path.splitext(input_path)[1]
        else:
            extension = ".mkv"

        return baseName + "".join(suffixes) + extension

    def _generate_output_path(
        self, config: TASConfig, output_dir: str, input_path: str
    ):
        """Generates appropriate output path based on input parameters."""
        if config.get_value("encode_method") == "png":
            output_name = self._generate_output_name(config, input_path)
            output_folder = os.path.join(output_dir, output_name)
            os.makedirs(output_folder, exist_ok=True)
            return os.path.join(output_folder, "frames_%05d.png")

        return os.path.join(output_dir, self._generate_output_name(config, input_path))

    def _get_video_files(self, input_paths: list[str]) -> list[str]:
        """Extract list of video files from input specification.

        Parameters
        ----------
        input_paths : list[str]
            A list of paths for each input.

        Returns
        -------
        list[str]
            A list of paths for all valid inputs. Invalid inputs are discarded.
        """
        output_paths = []
        for path in input_paths:
            if not os.path.exists(path):
                self.logger.warning(f"Path '{path}' not found. Skipping", pid=0)
                continue

            if os.path.isdir(path):
                for file in os.listdir(path):
                    ext = os.path.splitext(file)[1]
                    if ext.lower() in TASArgs.video_extensions:
                        output_paths.append(os.path.join(path, file))
            elif os.path.isfile(path):
                ext = os.path.splitext(path)[1].lower()
                if ext == ".txt":
                    with open(path) as file:
                        files = self._get_video_files(
                            [
                                line.strip()
                                for line in file.readlines()
                                if line.strip() != ""
                            ]
                        )
                        output_paths.extend(files)
                elif ext in TASArgs.video_extensions:
                    output_paths.append(path)
            # Handle semicolon-separated paths
            elif ";" in path:
                files = self._get_video_files(
                    [v.strip() for v in path.split(";") if v.strip() != ""]
                )
                output_paths.extend(files)
        return output_paths

    def get_input_files(self) -> list[PathConfiguration]:
        """Returns the paths of video files found at the input location."""
        config = TASConfig()
        output_dir = config.get_value("output")
        files = self._get_video_files([config.get_value("input")])

        input_count = len(files)
        self.logger.info(
            f"Found {input_count} video{'s' if input_count != 1 else ''} to process",
            pid=0,
        )
        self.logger.debug(f"Initializing output directory '{output_dir}'", pid=0)
        os.makedirs(output_dir, exist_ok=True)

        return [
            PathConfiguration(
                input_path=path,
                output_path=self._generate_output_path(
                    config,
                    output_dir,
                    path,
                ),
            )
            for path in files
        ]
