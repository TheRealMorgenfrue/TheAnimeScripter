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
        categpry_delim = ";"
        setting_delim = ","
        na = "null"  # Missing value

        arg_map = {"dedup": "", "sbd": "", "vfi": "", "sr": ""}

        if config.get_value("dedup", default=False):
            dedup = [
                f"dedup={config.get_value('dedup_method', default=na)}",
                f"sens={config.get_value('dedup_sens', default=na)}",
            ]
            arg_map["dedup"] = setting_delim.join(dedup)
        if config.get_value("sbd", default=False):
            scene_detect = [
                f"sbd={config.get_value('sbd_method', default=na)}",
                f"sens={config.get_value('sbd_sens', default=na)}",
            ]
            arg_map["sbd"] = setting_delim.join(scene_detect)
        if config.get_value("vfi", default=False):
            custom_vfi = config.get_value("custom_vfi_model", default="")
            vfi = [
                f"vfi={custom_vfi if custom_vfi else config.get_value('vfi_model', default=na)}",
                f"factor={config.get_value('vfi_factor', default=na)}",
                f"scale={config.get_value('vfi_scale', default=na)}",
                f"bsize={config.get_value('vfi_batch_size', default=na)}",
            ]
            ensemble = "ensemble" if config.get_value("ensemble", default=False) else ""
            dynamic_scale = (
                "dynamic_scale"
                if config.get_value("dynamic_scale", default=False)
                else ""
            )

            if ensemble:
                vfi.append(ensemble)
            if dynamic_scale:
                vfi.append(dynamic_scale)

            arg_map["vfi"] = setting_delim.join(vfi)
        if config.get_value("sr", default=False):
            sr = [
                f"sr={config.get_value('sr_model', default=na)}",
                f"factor={config.get_value('sr_factor', default=na)}",
            ]
            arg_map["sr"] = setting_delim.join(sr)

        #####

        # Handle URL input
        # if input_path in ["https://", "http://"]:
        #     return f"TAS-YTDLP-{random.randint(0, 1000)}.mp4"

        # Start with base name
        baseName = (
            os.path.splitext(os.path.basename(input_path))[0] if input_path else "TAS"
        )

        # Add processing indicators
        suffixes = [
            f"_dec={config.get_value('decode_method', default=na)}",
            f"enc={config.get_value('encode_method', default=na)}",
            f"dtype={config.get_value('precision', default=na)}",
        ]
        suffixes.extend([suffix for suffix in arg_map.values() if suffix])

        # Add random number to prevent overwrites
        file_id = f"_ID{random.randint(0, 1000)}"

        # Determine extension
        if (
            config.get_value("segment", default=False)
            or config.get_value("encode_method", default="") == "prores"
        ):
            extension = ".mov"
        elif config.get_value("encode_method") == "png":
            extension = ""
        elif input_path:
            extension = os.path.splitext(input_path)[1]
        else:
            extension = ".mkv"

        return baseName + categpry_delim.join(suffixes) + file_id + extension

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
        return sorted(output_paths)

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
