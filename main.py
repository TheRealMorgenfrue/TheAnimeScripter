"""
The Anime Scripter - AI Video Enhancement Toolkit

A high-performance AI video enhancement toolkit specialized for anime and general video content.
Provides professional-grade AI upscaling, interpolation, and restoration capabilities.

Copyright (C) 2023-present Nilas Tiago

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

import os
import sys
import traceback

from applib import CLIArguments, CoreApp, LoggingManager

from module.setup import process_arguments
from src.app.interfaces.mainwindow import TASMainWindow
from src.module.config.tas_config import TASConfig


def main():
    """
    Main entry point for The Anime Scripter application.

    Handles initialization, argument parsing, and coordinates video processing
    for single or multiple input files.
    """

    # Set application path
    os.environ["TAS_PATH"] = f"{os.path.dirname(os.path.abspath(__file__))}"

    if len(sys.argv) <= 1:
        # GUI mode
        CoreApp(TASMainWindow)
    else:
        # CLI mode
        logger = LoggingManager()

        if sys.platform == "win32":
            try:
                stream = "stdout"
                sys.stdout.reconfigure(encoding="utf-8")
                stream = "stderr"
                sys.stderr.reconfigure(encoding="utf-8")
            except Exception:
                logger.error(
                    f"Failed to reconfigure {stream}:\n{traceback.format_exc()}"
                )

        try:
            config = TASConfig()
            argument_handler = CLIArguments()
            arg_parser = argument_handler.create_argparser(config)
            args = arg_parser.parse_args()
            argument_handler.deserialize(args, config, merge=True)
            process_arguments(config)
        except KeyboardInterrupt:
            logger.warning("Process interrupted by user")
            # TODO: Cleanup here. Destroy running processes etc.
            sys.exit(0)
        except Exception:
            logger.critical(f"Fatal error in main execution\n{traceback.format_exc()}")
            # TODO: Cleanup here. Destroy running processes etc.
            sys.exit(1)


if __name__ == "__main__":
    main()
