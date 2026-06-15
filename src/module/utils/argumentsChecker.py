"""
Command Line Argument Parser

Handles parsing and validation of command line arguments for The Anime Scripter.
Provides comprehensive argument definitions and validation logic.
"""

import logging
import os
import sys

import src.module.constants as cs
from src.module.utils.logAndPrint import logAndPrint

from src.module.utils.cuda_checker import CudaChecker, detectGPUArchitecture

# class DidYouMeanArgumentParser(argparse.ArgumentParser):
#     """
#     Custom ArgumentParser that provides "did you mean?" suggestions for invalid choices.

#     When a user provides an invalid choice for an argument, this parser will
#     suggest similar valid choices based on string similarity.
#     """

#     def _levenshteinDistance(self, s1, s2):
#         """
#         Calculate the Levenshtein distance between two strings.

#         Args:
#             s1: First string
#             s2: Second string

#         Returns:
#             int: The edit distance between the two strings
#         """
#         if len(s1) < len(s2):
#             return self._levenshteinDistance(s2, s1)

#         if len(s2) == 0:
#             return len(s1)

#         previousRow = range(len(s2) + 1)
#         for i, c1 in enumerate(s1):
#             currentRow = [i + 1]
#             for j, c2 in enumerate(s2):
#                 insertions = previousRow[j + 1] + 1
#                 deletions = currentRow[j] + 1
#                 substitutions = previousRow[j] + (c1 != c2)
#                 currentRow.append(min(insertions, deletions, substitutions))
#             previousRow = currentRow

#         return previousRow[-1]

#     def similarityScore(self, invalidValue, choice):
#         """
#         Calculate a similarity score between an invalid value and a valid choice.
#         Higher score = better match.

#         Args:
#             invalidValue: The invalid value provided by the user
#             choice: A valid choice to compare against

#         Returns:
#             float: Similarity score (higher is better)
#         """
#         invalidLower = invalidValue.lower()
#         choiceLower = choice.lower()

#         distance = self._levenshteinDistance(invalidLower, choiceLower)
#         maxLen = max(len(invalidLower), len(choiceLower))
#         baseSimilarity = 1.0 - (distance / maxLen) if maxLen > 0 else 0

#         bonus = 0.0

#         if choiceLower.startswith(invalidLower) or invalidLower.startswith(choiceLower):
#             bonus += 0.4

#         commonPrefixLen = 0
#         for i in range(min(len(invalidLower), len(choiceLower))):
#             if invalidLower[i] == choiceLower[i]:
#                 commonPrefixLen += 1
#             else:
#                 break
#         bonus += (commonPrefixLen / max(len(invalidLower), len(choiceLower))) * 0.2

#         if invalidLower in choiceLower:
#             bonus += 0.15

#         return baseSimilarity + bonus

#     def getSuggestions(self, invalidValue, validChoices, maxSuggestions=5):
#         """
#         Get suggestions for an invalid value from a list of valid choices.

#         Args:
#             invalidValue: The invalid value provided by the user
#             validChoices: List of valid choices
#             maxSuggestions: Maximum number of suggestions to return

#         Returns:
#             List of suggested valid choices, sorted by similarity
#         """
#         # Score all choices
#         scoredChoices = [
#             (choice, self.similarityScore(invalidValue, choice))
#             for choice in validChoices
#         ]

#         # Sort by score (descending) and then by length (ascending) for ties
#         scoredChoices.sort(key=lambda x: (-x[1], len(x[0])))

#         # Return top suggestions (only if they have a reasonable score)
#         suggestions = []
#         for choice, score in scoredChoices:
#             if score > 0.3 and len(suggestions) < maxSuggestions:  # Minimum threshold
#                 suggestions.append(choice)

#         return suggestions

#     def error(self, message):
#         """
#         Override error method to provide "did you mean?" suggestions.

#         Args:
#             message: The original error message from argparse
#         """
#         # Check if this is an invalid choice error
#         if "invalid choice:" in message and "choose from" in message:
#             import re

#             from rich.console import Console
#             from rich.text import Text

#             # Pattern to match: argument --arg_name: invalid choice: 'value' (choose from choice1, choice2, ...)
#             # Note: choices are NOT quoted in the actual argparse output
#             match = re.search(
#                 r"argument (--?[\w-]+):\s*invalid choice:\s*'([^']+)'\s*\(choose from\s*(.+)\)",
#                 message,
#             )

#             if match:
#                 argName = match.group(1)
#                 invalidValue = match.group(2)
#                 choicesStr = match.group(3)

#                 # Parse the choices - they are comma-separated, possibly with spaces
#                 choices = [c.strip() for c in choicesStr.split(",") if c.strip()]

#                 if choices:
#                     suggestions = self.getSuggestions(invalidValue, choices)

#                     # Build enhanced error message with colors
#                     console = Console(stderr=True)

#                     # Print usage line
#                     self.print_usage(sys.stderr)

#                     # Build colored error message
#                     errorText = Text()
#                     errorText.append("main.py: error: ", style="bold red")
#                     errorText.append(f"argument {argName}: invalid choice: ")
#                     errorText.append(f"'{invalidValue}'", style="bold red")

#                     console.print(errorText)

#                     if suggestions:
#                         suggestionText = Text()
#                         suggestionText.append("\n  Did you mean: ", style="bold yellow")
#                         suggestionText.append(
#                             ", ".join(repr(s) for s in suggestions), style="bold green"
#                         )
#                         console.print(suggestionText)

#                     # Show first 10 valid choices in a single line
#                     choicesText = Text()
#                     choicesText.append("\n  Valid choices: ", style="bold cyan")
#                     displayedChoices = choices[:10]
#                     choicesStrDisplay = ", ".join(repr(c) for c in displayedChoices)
#                     if len(choices) > 10:
#                         choicesStrDisplay += f", ... ({len(choices) - 10} more)"
#                     choicesText.append(choicesStrDisplay, style="dim")
#                     console.print(choicesText)

#                     sys.exit(2)
#                     return

#         # For other errors, use the default behavior
#         super().error(message)


# def isAnyOtherProcessingMethodEnabled(args):
#     """
#     Check if any video processing operations are enabled.

#     Args:
#         args: Parsed command line arguments

#     Returns:
#         bool: True if any processing method is enabled
#     """
#     return any(
#         [
#             args.interpolate,
#             args.upscale,
#             args.segment,
#             args.restore,
#             args.sharpen,
#             args.resize,
#             args.dedup,
#             args.depth,
#             args.autoclip,
#             args.obj_detect,
#         ]
#     )


def argumentsChecker(args, outputPath, parser):

    _configureProcessingSettings(args)

    # Check CUDA availability and adjust methods if needed
    _adjustMethodsBasedOnCuda(args)

    if args.encode_method in ["gif", "png"]:
        logging.info(
            f"Encoding method is set to {args.encode_method}, disabling audio processing"
        )
        cs.AUDIO = False

    if not args.input:
        logging.error("No input specified")
        sys.exit()
    elif args.input.startswith(("http", "www")):
        processURL(args, outputPath)
    elif args.input.lower().endswith((".png", ".jpg", ".jpeg")):
        if "%" in args.input:
            logging.info(f"Image sequence pattern detected: {args.input}")
            args.input = os.path.abspath(args.input)
            cs.AUDIO = False
        else:
            raise Exception(
                "Single image input is not supported. For image sequences, use a pattern like 'frames_%05d.png' or provide a folder containing PNG files."
            )
    elif args.input.lower().endswith(".gif"):
        if args.encode_method != "gif":
            logging.error(
                "GIF input detected but encoding method is not set to GIF, defaulting to GIF encoding"
            )
            args.encode_method = "gif"
    else:
        try:
            args.input = os.path.abspath(args.input)
        except Exception:
            logging.error("Error processing input")
            sys.exit()

    return args


def _configureProcessingSettings(args):
    if args.slowmo:
        cs.AUDIO = False
        logging.info("Slow motion enabled, audio processing disabled")

    if args.static_step and isinstance(args.interpolate_factor, float):
        logging.info("Interpolate Factor is a float, static step will be disabled")
        args.static_step = False

    if args.dedup:
        if not args.smooth_dedup:
            cs.AUDIO = False
            logging.info(
                "Deduplication enabled and smooth dedup disabled, audio processing disabled"
            )

        if args.dedup_method in ["ssim", "ssim-cuda"]:
            args.dedup_sens = 1.0 - (args.dedup_sens / 1000)
        elif args.dedup_method in ["flownets"]:
            args.dedup_sens = args.dedup_sens / 100

        logging.info(
            f"New dedup sensitivity for {args.dedup_method} is: {args.dedup_sens}"
        )

    if args.sharpen:
        args.sharpen_sens = args.sharpen_sens / 100
        logging.info(f"New sharpen sensitivity is: {args.sharpen_sens}")

    if args.autoclip:
        # For some reason, the sensitivity is inverted in the autoclip method, could be some hard math that I don't understand
        # but for now, we will just invert it to make it work as expected
        args.autoclip_sens = float(100 - args.autoclip_sens)
        logging.info(f"New autoclip sensitivity is: {args.autoclip_sens}")


def _adjustMethodsBasedOnCuda(args):
    def adjustMethod(method, modelsList):
        base = method.lower().split("-")[0]
        directML = f"{base}-directml"
        if directML in modelsList:
            return directML
        newMethod = f"{base}-ncnn"
        if newMethod in modelsList:
            return newMethod
        return method

    isCuda = CudaChecker()

    # Check if GPU architecture supports modern CUDA features
    needsFallback = False
    if isCuda.cuda_available:
        isModernGPU, gpuName, computeCap = detectGPUArchitecture()
        if not isModernGPU:
            logAndPrint(
                f"Detected {gpuName} (compute capability: {computeCap}). "
                f"This GPU may not support modern CUDA kernels. "
                f"Automatically switching to DirectML/NCNN backends for compatibility.",
                "yellow",
            )
            needsFallback = True
    else:
        needsFallback = True

    if needsFallback:
        from .downloadModels import modelsList

        availableModels = modelsList()
        methodAttributes = [
            "interpolate_method",
            "upscale_method",
            "segment_method",
            "depth_method",
            "restore_method",
            "dedup_method",
            "obj_detect_method",
        ]

        methodToFlag = {
            "interpolate_method": "interpolate",
            "upscale_method": "upscale",
            "segment_method": "segment",
            "depth_method": "depth",
            "restore_method": "restore",
            "dedup_method": "dedup",
            "obj_detect_method": "obj_detect",
        }

        for attr in methodAttributes:
            flagName = methodToFlag.get(attr)
            if flagName and not getattr(args, flagName):
                continue

            currentMethod = getattr(args, attr)

            if attr == "restore_method" and isinstance(currentMethod, list):
                adjusted = []
                for method in currentMethod:
                    if any(
                        backend in method.lower()
                        for backend in ["-directml", "-ncnn", "-tensorrt"]
                    ):
                        logging.info(
                            f"{attr} method {method} already using non-default backend"
                        )
                        adjusted.append(method)
                        continue

                    newMethod = adjustMethod(method, availableModels)
                    if newMethod != method:
                        logging.info(
                            f"Adjusted {attr} method from {method} to {newMethod}"
                        )
                    adjusted.append(newMethod)
                setattr(args, attr, adjusted)
            else:
                if any(
                    backend in currentMethod.lower()
                    for backend in ["-directml", "-ncnn", "-tensorrt"]
                ):
                    logging.info(
                        f"{attr} already using non-default backend: {currentMethod}"
                    )
                    continue

                newMethod = adjustMethod(currentMethod, availableModels)
                if newMethod != currentMethod:
                    logging.info(f"Adjusted {attr} from {currentMethod} to {newMethod}")
                    setattr(args, attr, newMethod)
                else:
                    logging.info(
                        f"No adjustment for {attr} ({currentMethod} remains unchanged)"
                    )
