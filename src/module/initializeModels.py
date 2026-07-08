"""
Model Initialization and Processing Functions

This module handles the initialization and execution of various AI models
for video processing operations including object detection, auto-clipping,
segmentation, depth estimation, and the main processing pipeline.
"""

import logging

import torch


def autoClip(self):
    """
    Initialize and execute automatic scene detection and clipping.

    Args:
        self: VideoProcessor instance containing processing parameters
    """
    from src.module.autoclip.autoclip import AutoClip

    AutoClip(
        self.input,
        self.autoclipSens,
        self.inpoint,
        self.outpoint,
    )


def initialize_models(width: int, height: int):
    """
    Initialize all AI models for the video processing pipeline.

    Args:
        self: VideoProcessor instance containing processing parameters

    Returns:
        tuple: Contains output dimensions and initialized processing functions
            - outputWidth (int): Final output video width
            - outputHeight (int): Final output video height
            - upscaleProcess: Upscaling model function or None
            - interpolateProcess: Interpolation model function or None
            - restoreProcess: Restoration model function or None
            - dedupProcess: Deduplication function or None
    """
    outputWidth = self.width
    outputHeight = self.height
    upscaleProcess = None
    interpolateProcess = None
    restoreProcess = None
    dedupProcess = None

    if self.upscale:
        from src.module.unifiedUpscale import UniversalPytorch

        outputWidth *= self.upscaleFactor
        outputHeight *= self.upscaleFactor
        logging.info(f"Upscaling to {outputWidth}x{outputHeight}")
        match self.upscaleMethod:
            case (
                "shufflecugan"
                | "compact"
                | "ultracompact"
                | "superultracompact"
                | "span"
                | "open-proteus"
                | "aniscale2"
                | "shufflespan"
                | "rtmosr"
                | "saryn"
                | "fallin_soft"
                | "fallin_strong"
                | "gauss"
            ):
                upscaleProcess = UniversalPytorch(
                    self.upscaleMethod,
                    self.upscaleFactor,
                    self.half,
                    self.width,
                    self.height,
                    self.customModel,
                    self.compileMode,
                )

            case (
                "compact-directml"
                | "ultracompact-directml"
                | "superultracompact-directml"
                | "span-directml"
                | "open-proteus-directml"
                | "aniscale2-directml"
                | "shufflespan-directml"
                | "shufflecugan-directml"
                | "rtmosr-directml"
                | "saryn-directml"
                | "fallin_soft-directml"
                | "fallin_strong-directml"
                | "compact-openvino"
                | "ultracompact-openvino"
                | "superultracompact-openvino"
                | "span-openvino"
                | "open-proteus-openvino"
                | "aniscale2-openvino"
                | "shufflespan-openvino"
                | "shufflecugan-openvino"
                | "rtmosr-openvino"
                | "saryn-openvino"
                | "fallin_soft-openvino"
                | "fallin_strong-openvino"
                | "gauss-openvino"
                | "gauss-directml"
            ):
                from src.module.unifiedUpscale import UniversalDirectML

                upscaleProcess = UniversalDirectML(
                    self.upscaleMethod,
                    self.upscaleFactor,
                    self.half,
                    self.width,
                    self.height,
                    self.customModel,
                )

            case "animesr-openvino" | "animesr-directml":
                from src.module.unifiedUpscale import AnimeSRDirectML

                upscaleProcess = AnimeSRDirectML(
                    self.upscaleMethod,
                    self.half,
                    self.width,
                    self.height,
                )

            case "shufflecugan-ncnn" | "span-ncnn":
                from src.module.unifiedUpscale import UniversalNCNN

                upscaleProcess = UniversalNCNN(
                    self.upscaleMethod,
                    self.upscaleFactor,
                )

            case (
                "shufflecugan-tensorrt"
                | "compact-tensorrt"
                | "ultracompact-tensorrt"
                | "superultracompact-tensorrt"
                | "span-tensorrt"
                | "open-proteus-tensorrt"
                | "aniscale2-tensorrt"
                | "shufflespan-tensorrt"
                | "rtmosr-tensorrt"
                | "saryn-tensorrt"
                | "fallin_soft-tensorrt"
                | "fallin_strong-tensorrt"
                | "gauss-tensorrt"
            ):
                from src.module.unifiedUpscale import UniversalTensorRT

                upscaleProcess = UniversalTensorRT(
                    self.upscaleMethod,
                    self.upscaleFactor,
                    self.half,
                    self.width,
                    self.height,
                    self.customModel,
                    self.forceStatic,
                )

            case "animesr":
                from src.module.unifiedUpscale import AnimeSR

                upscaleProcess = AnimeSR(
                    2,
                    self.half,
                    self.width,
                    self.height,
                    self.compileMode,
                )

            case "animesr-tensorrt":
                from src.module.unifiedUpscale import AnimeSRTensorRT

                upscaleProcess = AnimeSRTensorRT(
                    2,
                    self.half,
                    self.width,
                    self.height,
                )
    if self.interpolate:
        logging.info(
            f"Interpolating from {format(self.fps, '.3f')}fps to {format(self.fps * self.interpolateFactor, '.3f')}fps"
        )
        match self.interpolateMethod:
            case (
                "rife"
                | "rife4.6"
                | "rife4.15-lite"
                | "rife4.16-lite"
                | "rife4.17"
                | "rife4.18"
                | "rife4.20"
                | "rife4.21"
                | "rife4.22"
                | "rife4.22-lite"
                | "rife4.25"
                | "rife4.25-lite"
                | "rife_elexor"
                | "rife4.25-heavy"
            ):
                from src.module.unifiedInterpolate import RifeCuda

                interpolateProcess = RifeCuda(
                    self.half,
                    self.width,
                    self.height,
                    self.interpolateMethod,
                    self.ensemble,
                    self.interpolateFactor,
                    self.dynamicScale,
                    self.staticStep,
                    compileMode=self.compileMode,
                )

            case "rife4.25-depth":
                from src.module.unifiedInterpolate import DepthGuidedRifeCuda

                interpolateProcess = DepthGuidedRifeCuda(
                    width=self.width,
                    height=self.height,
                    half=self.half,
                    interpolate_method="rife4.25",
                    depth_method=self.depthMethod,
                    depth_quality=self.depthQuality,
                    ensemble=self.ensemble,
                )

            case (
                "rife-ncnn"
                | "rife4.6-ncnn"
                | "rife4.15-lite-ncnn"
                | "rife4.16-lite-ncnn"
                | "rife4.17-ncnn"
                | "rife4.18-ncnn"
                | "rife4.20-ncnn"
                | "rife4.21-ncnn"
                | "rife4.22-ncnn"
                | "rife4.22-lite-ncnn"
            ):
                from src.module.unifiedInterpolate import RifeNCNN

                interpolateProcess = RifeNCNN(
                    self.interpolateMethod,
                    self.ensemble,
                    self.width,
                    self.height,
                    self.half,
                    self.interpolateFactor,
                )

            case (
                "rife-tensorrt"
                | "rife4.6-tensorrt"
                | "rife4.15-tensorrt"
                | "rife4.15-lite-tensorrt"
                | "rife4.17-tensorrt"
                | "rife4.18-tensorrt"
                | "rife4.20-tensorrt"
                | "rife4.21-tensorrt"
                | "rife4.22-tensorrt"
                | "rife4.22-lite-tensorrt"
                | "rife4.25-tensorrt"
                | "rife4.25-lite-tensorrt"
                | "rife_elexor-tensorrt"
                | "rife4.25-heavy-tensorrt"
            ):
                from src.module.unifiedInterpolate import RifeTensorRT

                interpolateProcess = RifeTensorRT(
                    self.interpolateMethod,
                    self.interpolateFactor,
                    self.width,
                    self.height,
                    self.half,
                    self.ensemble,
                )

            case "gmfss":
                from module.models.vfi.gmfss.gmfss import GMFSS

                interpolateProcess = GMFSS(
                    int(self.interpolateFactor),
                    self.half,
                    outputWidth,
                    outputHeight,
                    self.ensemble,
                    compileMode=self.compileMode,
                )

            case "gmfss-tensorrt":
                from src.module.gmfss import GMFSSTensorRT

                interpolateProcess = GMFSSTensorRT(
                    int(self.interpolateFactor),
                    outputWidth,
                    outputHeight,
                    self.half,
                    self.ensemble,
                )

            case (
                "rife4.6-directml"
                | "rife4.6-openvino"
                | "rife4.15-directml"
                | "rife4.17-directml"
                | "rife4.18-directml"
                | "rife4.20-directml"
                | "rife4.21-directml"
                | "rife4.22-directml"
                | "rife4.22-lite-directml"
                | "rife4.25-directml"
                | "rife4.25-lite-directml"
                | "rife4.25-heavy-directml"
                | "rife4.15-openvino"
                | "rife4.17-openvino"
                | "rife4.18-openvino"
                | "rife4.20-openvino"
                | "rife4.21-openvino"
                | "rife4.22-openvino"
                | "rife4.22-lite-openvino"
                | "rife4.25-openvino"
                | "rife4.25-lite-openvino"
                | "rife4.25-heavy-openvino"
            ):
                from src.module.unifiedInterpolate import RifeDirectML

                interpolateProcess = RifeDirectML(
                    self.interpolateMethod,
                    self.interpolateFactor,
                    self.width,
                    self.height,
                    self.half,
                    self.ensemble,
                )

            case "distildrba" | "distildrba-lite":
                from src.module.unifiedInterpolate import DistilDRBACuda

                interpolateProcess = DistilDRBACuda(
                    self.half,
                    self.width,
                    self.height,
                    self.interpolateMethod,
                    interpolateFactor=self.interpolateFactor,
                    compileMode=self.compileMode,
                )

            case "atr":
                from src.module.unifiedInterpolate import ATRCuda

                interpolateProcess = ATRCuda(
                    self.half,
                    self.width,
                    self.height,
                    self.interpolateMethod,
                    interpolateFactor=self.interpolateFactor,
                    compileMode=self.compileMode,
                )

            case "distildrba-lite-tensorrt" | "distildrba-tensorrt":
                from src.module.unifiedInterpolate import DistilDRBATensorRT

                interpolateProcess = DistilDRBATensorRT(
                    self.half,
                    self.width,
                    self.height,
                    self.interpolateMethod,
                    interpolateFactor=self.interpolateFactor,
                )

    if self.restore:
        if ADOBE:
            progressState.update(
                {"status": f"Initializing restore model: {self.restoreMethod}..."}
            )

        restoreMethods = (
            self.restoreMethod
            if isinstance(self.restoreMethod, list)
            else [self.restoreMethod]
        )
        restoreProcesses = []

        for method in restoreMethods:
            match method:
                case (
                    "scunet"
                    | "dpir"
                    | "nafnet"
                    | "real-plksr"
                    | "anime1080fixer"
                    | "gater3"
                    | "deh264_real"
                    | "deh264_span"
                    | "hurrdeblur"
                    | "dehalo"
                    | "scunet-openvino"
                    | "anime1080fixer-openvino"
                    | "gater3-openvino"
                    | "deh264_real-openvino"
                    | "deh264_span-openvino"
                    | "hurrdeblur-openvino"
                    | "dehalo-openvino"
                ):
                    from src.module.unifiedRestore import UnifiedRestoreCuda

                    restoreProcesses.append(
                        UnifiedRestoreCuda(
                            method,
                            self.half,
                        )
                    )

                case (
                    "anime1080fixer-tensorrt"
                    | "gater3-tensorrt"
                    | "scunet-tensorrt"
                    | "codeformer-tensorrt"
                    | "deh264_real-tensorrt"
                    | "deh264_span-tensorrt"
                    | "hurrdeblur-tensorrt"
                    | "dehalo-tensorrt"
                ):
                    from src.module.unifiedRestore import UnifiedRestoreTensorRT

                    restoreProcesses.append(
                        UnifiedRestoreTensorRT(
                            method,
                            self.half,
                            self.width,
                            self.height,
                            self.forceStatic,
                        )
                    )

                case (
                    "anime1080fixer-directml"
                    | "anime1080fixer-openvino"
                    | "gater3-directml"
                    | "scunet-directml"
                    | "codeformer-directml"
                    | "deh264_real-directml"
                    | "deh264_span-directml"
                    | "hurrdeblur-directml"
                    | "dehalo-directml"
                ):
                    from src.module.unifiedRestore import UnifiedRestoreDirectML

                    restoreProcesses.append(
                        UnifiedRestoreDirectML(
                            method,
                            self.half,
                            self.width,
                            self.height,
                        )
                    )
                case "fastlinedarken":
                    from module.models.extraArches.fastlinedarken import (
                        FastLineDarkenWithStreams,
                    )

                    restoreProcesses.append(
                        FastLineDarkenWithStreams(
                            self.half,
                        )
                    )
                case "fastlinedarken-tensorrt":
                    from module.models.extraArches.fastlinedarken import (
                        FastLineDarkenTRT,
                    )

                    restoreProcesses.append(
                        FastLineDarkenTRT(
                            self.half,
                            self.height,
                            self.width,
                        )
                    )

                case (
                    "linethinner-lite"
                    | "linethinner-medium"
                    | "linethinner-heavy"
                    | "linethinner-lite-cuda"
                    | "linethinner-medium-cuda"
                    | "linethinner-heavy-cuda"
                ):
                    from module.models.extraArches.linethinner import LineThin

                    device = "cuda" if "cuda" in method else "cpu"
                    variant = method.replace("-cuda", "").replace("linethinner-", "")

                    restoreProcesses.append(
                        LineThin(
                            variant=variant,
                            half=self.half,
                            device=device,
                        )
                    )

        if len(restoreProcesses) == 1:
            restoreProcess = restoreProcesses[0]
        else:
            restoreProcess = RestoreChain(restoreProcesses)

    if self.dedup:
        if ADOBE:
            progressState.update(
                {"status": f"Initializing deduplication: {self.dedupMethod}..."}
            )

        match self.dedupMethod:
            case "ssim":
                from module.models.dedup.dedup import DedupSSIM

                dedupProcess = DedupSSIM(
                    self.dedupSens,
                )

            case "mse":
                from module.models.dedup.dedup import DedupMSE

                dedupProcess = DedupMSE(
                    self.dedupSens,
                )

            case "ssim-cuda":
                from module.models.dedup.dedup import DedupSSIMCuda

                dedupProcess = DedupSSIMCuda(
                    self.dedupSens,
                    self.half,
                )

            case "vmaf" | "vmaf-cuda":
                from module.models.dedup.dedup import DedupVMAF

                dedupProcess = DedupVMAF(
                    dedupMethod=self.dedupMethod,
                    treshold=self.dedupSens,
                    half=self.half,
                )

            case "mse-cuda":
                from module.models.dedup.dedup import DedupMSECuda

                dedupProcess = DedupMSECuda(
                    self.dedupSens,
                    self.half,
                )

            case "flownets":
                from module.models.dedup.dedup import DedupFlownetS

                dedupProcess = DedupFlownetS(
                    half=self.half,
                    dedupSens=self.dedupSens,
                    height=self.height,
                    width=self.width,
                )

    return (
        outputWidth,
        outputHeight,
        upscaleProcess,
        interpolateProcess,
        restoreProcess,
        dedupProcess,
        restoreProcess,
        dedupProcess,
        restoreProcess,
        dedupProcess,
        dedupProcess,
        restoreProcess,
        dedupProcess,
        dedupProcess,
        restoreProcess,
        dedupProcess,
        dedupProcess,
        restoreProcess,
        dedupProcess,
    )
