import logging
import math
import os

import torch
from src.module.utils.downloadModels import downloadModels, weightsDir
from src.module.utils.logAndPrint import logAndPrint
from torch.nn import functional as F

from src.module.utils.cuda_checker import CudaChecker

checker = CudaChecker()

# from: https://github.com/HolyWu/vs-gmfss_fortuna/blob/master/vsgmfss_fortuna/__init__.py


class GMFSS:
    def __init__(
        self,
        interpolation_factor,
        half,
        width,
        height,
        ensemble=False,
        compileMode: str = "default",
    ):
        self.width = width
        self.height = height
        self.half = half
        self.interpolation_factor = interpolation_factor
        self.ensemble = ensemble
        self.compileMode: str = compileMode

        self.ph = ((self.height - 1) // 64 + 1) * 64
        self.pw = ((self.width - 1) // 64 + 1) * 64

        if self.width > 1920 or self.height > 1080:
            self.scale = 0.5
        else:
            self.scale = 1

        self.handleModel()

    def handleModel(self):
        if not os.path.exists(os.path.join(weightsDir, "gmfss")):
            modelDir = os.path.dirname(downloadModels("gmfss"))
        else:
            modelDir = os.path.join(weightsDir, "gmfss")

        modelType = "union"

        self.device = torch.device("cuda" if checker.cuda_available else "cpu")

        torch.set_grad_enabled(False)
        if checker.cuda_available:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

        from .model.GMFSS import GMFSS as Model

        self.model = Model(modelDir, modelType, self.scale, ensemble=self.ensemble)
        self.model.eval().to(self.device, memory_format=torch.channels_last)

        self.dtype = torch.float
        if checker.cuda_available and self.half:
            self.model.half()
            self.dtype = torch.half

        if self.compileMode != "default":
            try:
                if self.compileMode == "max":
                    self.model.compile(mode="max-autotune-no-cudagraphs")
                elif self.compileMode == "max-graphs":
                    self.model.compile(
                        mode="max-autotune-no-cudagraphs", fullgraph=True
                    )
            except Exception as e:
                logging.error(
                    f"Error compiling model {self.interpolateMethod} with mode {self.compileMode}: {e}"
                )
                logAndPrint(
                    f"Error compiling model {self.interpolateMethod} with mode {self.compileMode}: {e}",
                    "red",
                )

            self.compileMode = "default"

        self.I0 = torch.zeros(
            1,
            3,
            self.ph,
            self.pw,
            dtype=torch.float16 if self.half else torch.float32,
            device=self.device,
        )

        self.I1 = torch.zeros(
            1,
            3,
            self.ph,
            self.pw,
            dtype=torch.float16 if self.half else torch.float32,
            device=self.device,
        )

        self.stream = torch.cuda.Stream()
        self.firstRun = True

    @torch.inference_mode()
    def cacheFrame(self):
        self.I0.copy_(self.I1, non_blocking=True)
        # self.model.cacheFrame()

    @torch.inference_mode()
    def processFrame(self, frame):
        return frame.to(
            self.device,
            non_blocking=True,
            dtype=torch.float16 if self.half else torch.float32,
        ).to(memory_format=torch.channels_last)

    @torch.inference_mode()
    def padFrame(self, frame):
        return (
            F.pad(frame, [0, self.pw - self.width, 0, self.ph - self.height])
            if (self.pw != self.width or self.ph != self.height)
            else frame
        )

    @torch.inference_mode()
    def __call__(self, frame, interpQueue, framesToInsert: int = 2, timesteps=None):
        with torch.cuda.stream(self.stream):
            if self.firstRun is True:
                self.I0 = self.padFrame(self.processFrame(frame))
                self.firstRun = False
                return

            self.I1 = self.padFrame(self.processFrame(frame))

            for i in range(framesToInsert):
                if timesteps is not None and i < len(timesteps):
                    t = timesteps[i]
                else:
                    t = (i + 1) * 1 / (framesToInsert + 1)
                timestep = torch.tensor(
                    [t],
                    dtype=self.dtype,
                    device=checker.device,
                )
                output = self.model(self.I0, self.I1, timestep)[
                    :, :, : self.height, : self.width
                ]
                self.stream.synchronize()
                interpQueue.put(output)

            self.cacheFrame()
