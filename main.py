"""
The Anime Scripter - AI Video Enhancement Toolkit

A high-performance AI video enhancement toolkit specialized for anime and general video content.
Provides professional-grade AI upscaling, interpolation, and restoration capabilities.

Copyright (C) 2023-2025 Nilas Tiago
Copyright (C) 2026-present TheRealMorgenfrue

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see {http://www.gnu.org/licenses/}.

Home: https://github.com/NevermindNilas/TheAnimeScripter
"""

import platform
import warnings

# Pydantic warns about "model" being a reserved name
warnings.filterwarnings("ignore", module="pydantic")

import os  # noqa: I001
import sys
import traceback


def _patch_nvidia_library_path():
    """Automatically add nvidia library paths to LD_LIBRARY_PATH on Linux when using CUDAExecutionProvider.

    ONNX Runtime fails to load due to missing libcudnn.so.9: https://github.com/microsoft/onnxruntime/issues/25609.
    Patch from: https://github.com/microsoft/onnxruntime/pull/25628

    """
    # Only apply this patch on Linux systems
    if platform.system() != "Linux":
        return

    # Check if we've already patched the path to avoid doing it multiple times
    if os.environ.get("ORT_NVIDIA_PATH_PATCHED") == "1":
        return

    try:
        # Try to find the nvidia package
        import importlib.util

        spec = importlib.util.find_spec("nvidia")
        if spec is None or not spec.submodule_search_locations:
            return

        # Get the nvidia package path
        nvidia_path = spec.submodule_search_locations[0]

        # Check for cudnn library path
        import pathlib

        cudnn_lib_path = pathlib.Path(nvidia_path) / "cudnn" / "lib"
        if not cudnn_lib_path.exists():
            return

        # Convert to string path
        cudnn_lib_str = str(cudnn_lib_path.resolve())

        # Get current LD_LIBRARY_PATH
        current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")

        # If path is not already in LD_LIBRARY_PATH, add it
        if cudnn_lib_str not in current_ld_path.split(":"):
            if current_ld_path:
                os.environ["LD_LIBRARY_PATH"] = f"{cudnn_lib_str}:{current_ld_path}"
            else:
                os.environ["LD_LIBRARY_PATH"] = cudnn_lib_str

            # Mark that we've patched the path
            os.environ["ORT_NVIDIA_PATH_PATCHED"] = "1"

            # Restart the process to make sure the library loading takes effect
            # Only do this if we're importing the main onnxruntime package (not in a subprocess)
            if __name__ == "__main__" or (__package__ and "onnxruntime" in __package__):
                os.execv(sys.executable, [sys.executable] + sys.argv)

    except Exception:
        # If anything goes wrong, just continue without patching
        # This is a best-effort approach
        pass


_patch_nvidia_library_path()


import torch  # Initialize torch before any other module in TAS. Required for NeLux  # noqa: F401, I001
import onnxruntime  # noqa: F401, I001 # Import after PyTorch. See https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#compatibility-with-pytorch

from applib import CLIArguments, CoreApp, LoggingManager

from src.app.interfaces.mainwindow import TASMainWindow
from src.module.config.tas_args import TASArgs
from src.module.config.tas_config import TASConfig
from src.module.setup import process_arguments


def main():
    """
    Main entry point for The Anime Scripter application.
    """
    # Set application path
    os.environ["TAS_PATH"] = f"{os.path.dirname(os.path.abspath(__file__))}"

    if len(sys.argv) <= 1:
        # GUI mode
        CoreApp(TASMainWindow)
    else:
        # CLI mode
        logger = LoggingManager()

        try:
            args = TASArgs()
            config = TASConfig()
            argument_handler = CLIArguments()
            arg_parser = argument_handler.create_argparser(
                config.template,
                sync_with_config=config,
                name=args.name,
                version=args._core_app_version,
            )
            args = arg_parser.parse_args()
            argument_handler.deserialize(args, config, merge=True)
            process_arguments()
        except KeyboardInterrupt:
            logger.warning("Process interrupted by user")
            # TODO: Cleanup here. Destroy running processes etc.
            sys.exit(0)
        except Exception:
            logger.critical(
                f"Fatal error in main execution\n{traceback.format_exc()}",
                gui=True,
                pid=0,
            )
            # TODO: Cleanup here. Destroy running processes etc.
            sys.exit(1)


if __name__ == "__main__":
    main()
