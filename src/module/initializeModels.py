import torch

from module.config.tas_config import TASConfig


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
    config = TASConfig()

    if self.interpolate:
        match self.interpolateMethod:
            case "rife4.6" | "rife_elexor":
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

            case "gmfss":
                from src.module.models.vfi.gmfss.gmfss import GMFSS

                interpolateProcess = GMFSS(
                    int(self.interpolateFactor),
                    self.half,
                    outputWidth,
                    outputHeight,
                    self.ensemble,
                    compileMode=self.compileMode,
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
