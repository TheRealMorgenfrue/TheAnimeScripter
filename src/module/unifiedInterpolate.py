import logging
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from .utils.cuda_checker import CudaChecker

checker = CudaChecker()

torch.set_float32_matmul_precision("medium")


def importRifeArch(interpolateMethod, version):
    match version:
        case "v1":
            match interpolateMethod:
                case "rife4.25-heavy":
                    from .models.vfi.rife.IFNet_rife425heavy import IFNet
                case "rife4.25-lite":
                    from .models.vfi.rife.IFNet_rife425lite import IFNet
                case "rife" | "rife4.25":
                    from .models.vfi.rife.IFNet_rife425 import IFNet
                case "rife4.22-lite":
                    from .models.vfi.rife.IFNet_rife422lite import IFNet
                case "rife4.22":
                    from .models.vfi.rife.IFNet_rife422 import IFNet
                case "rife4.21":
                    from .models.vfi.rife.IFNet_rife421 import IFNet
                case "rife4.20":
                    from .models.vfi.rife.IFNet_rife420 import IFNet
                case "rife4.18":
                    from .models.vfi.rife.IFNet_rife418 import IFNet
                case "rife4.17":
                    from .models.vfi.rife.IFNet_rife417 import IFNet
                case "rife4.15-lite":
                    from .models.vfi.rife.IFNet_rife415lite import IFNet
                case "rife4.16-lite":
                    from .models.vfi.rife.IFNet_rife416lite import IFNet
                case "rife4.6":
                    from .models.vfi.rife.IFNet_rife46 import IFNet
                case "rife_elexor":
                    from .models.vfi.rife.IFNet_elexor_cuda import IFNet
            return IFNet

        case "v3":
            match interpolateMethod:
                case "rife4.25-heavy-tensorrt":
                    from .models.vfi.rife.Rife425_heavy_v3 import IFNet

                    Head = True

                case "rife4.25-lite-tensorrt":
                    from .models.vfi.rife.Rife425_lite_v3 import IFNet

                    Head = True
                case "rife4.25-tensorrt":
                    from .models.vfi.rife.Rife425_v3 import IFNet

                    Head = True
                case "rife4.22-tensorrt":
                    from .models.vfi.rife.Rife422_v3 import IFNet

                    Head = True
                case "rife4.22-lite-tensorrt":
                    from .models.vfi.rife.Rife422_lite_v3 import IFNet

                    Head = True
                case "rife4.21-tensorrt":
                    from .models.vfi.rife.Rife422_v3 import IFNet

                    Head = True
                case "rife4.20-tensorrt":
                    from .models.vfi.rife.Rife420_v3 import IFNet

                    Head = True
                case "rife4.18-tensorrt":
                    from .models.vfi.rife.Rife415_v3 import IFNet

                    Head = True
                case "rife4.17-tensorrt":
                    from .models.vfi.rife.Rife415_v3 import IFNet

                    Head = True
                case "rife4.15-tensorrt":
                    from .models.vfi.rife.Rife415_v3 import IFNet

                    Head = True
                case "rife4.6-tensorrt":
                    from .models.vfi.rife.Rife46_v3 import IFNet

                    Head = False
                case "rife4.6-directml" | "rife4.6-openvino":
                    from .models.vfi.rife.Rife_directml import IFNet_46 as IFNet

                    Head = False
                case "rife4.22-directml" | "rife4.22-openvino":
                    from .models.vfi.rife.Rife_directml import IFNet_422 as IFNet

                    Head = True
                case (
                    "rife4.15-directml"
                    | "rife4.17-directml"
                    | "rife4.18-directml"
                    | "rife4.15-openvino"
                    | "rife4.17-openvino"
                    | "rife4.18-openvino"
                ):
                    from .models.vfi.rife.Rife_directml import IFNet_415 as IFNet

                    Head = True
                case (
                    "rife4.20-directml"
                    | "rife4.21-directml"
                    | "rife4.20-openvino"
                    | "rife4.21-openvino"
                ):
                    from .models.vfi.rife.Rife_directml import IFNet_420 as IFNet

                    Head = True
                case "rife4.22-lite-directml" | "rife4.22-lite-openvino":
                    from .models.vfi.rife.Rife_directml import IFNet_422_lite as IFNet

                    Head = True
                case "rife4.25-directml" | "rife4.25-openvino":
                    from .models.vfi.rife.Rife_directml import IFNet_425 as IFNet

                    Head = True
                case "rife4.25-lite-directml" | "rife4.25-lite-openvino":
                    from .models.vfi.rife.Rife_directml import IFNet_425_lite as IFNet

                    Head = True
                case "rife4.25-heavy-directml" | "rife4.25-heavy-openvino":
                    from .models.vfi.rife.Rife_directml import IFNet_425_heavy as IFNet

                    Head = True
                case "rife_elexor-tensorrt":
                    from .models.vfi.rife.IFNet_elexor import IFNet

                    Head = True
            return IFNet, Head


class RifeCuda:
    def __init__(
        self,
        half,
        width,
        height,
        interpolateMethod,
        ensemble=False,
        interpolateFactor=2,
        dynamicScale=False,
        staticStep=False,
        compileMode: str = "default",
    ):
        """
        Initialize the RIFE model

        Args:
            half (bool): Half resolution
            width (int): Width of the frame
            height (int): Height of the frame
            interpolateMethod (str): Interpolation method
            ensemble (bool, optional): Ensemble. Defaults to False.
            interpolateFactor (int, optional): Interpolation factor. Defaults to 2.
            dynamicScale (bool, optional): Use Dynamic scale. Defaults to False.
            staticStep (bool, optional): Use static timestep. Defaults to False.
        """
        self.half = half
        self.scale = 1.0
        self.width = width
        self.height = height
        self.interpolateMethod = interpolateMethod
        self.ensemble = ensemble
        self.interpolateFactor = interpolateFactor
        self.dynamicScale = dynamicScale
        self.staticStep = staticStep
        self.compileMode = compileMode

        if self.width > 1920 and self.height > 1080:
            self.scale = 0.5
            if self.half:
                logAndPrint(
                    "UHD and fp16 are not compatible with RIFE, defaulting to fp32",
                    "yellow",
                )
                self.half = False

        self.handleModel()

    def handleModel(self):
        """
        Load the desired model
        """
        if ADOBE:
            progressState.update(
                {
                    "status": f"Loading RIFE interpolation model: {self.interpolateMethod}..."
                }
            )

        self.filename = modelsMap(self.interpolateMethod)
        if not os.path.exists(os.path.join(weightsDir, "rife", self.filename)):
            modelPath = downloadModels(model=self.interpolateMethod)
        else:
            modelPath = os.path.join(weightsDir, "rife", self.filename)

        self.dType = torch.float16 if self.half else torch.float32

        IFNet = importRifeArch(self.interpolateMethod, "v1")
        if self.interpolateMethod in ["rife_elexor"] and self.staticStep:
            self.staticStep = False
            logAndPrint(
                "Static step is not supported for rife_elexor, automatically disabling it",
                "yellow",
            )
        if (
            self.interpolateMethod not in ["rife4.6", "rife4.15", "rife4.15-lite"]
            and self.staticStep
        ):
            self.staticStep = False
            logAndPrint(
                "Static step is not supported for this interpolation model yet, automatically disabling it",
                "yellow",
            )
        if self.interpolateMethod in ["rife_elexor"]:
            self.model = IFNet(
                self.scale,
                self.ensemble,
                self.dType,
                checker.device,
                self.width,
                self.height,
                self.interpolateFactor,
            )
        else:
            if self.interpolateMethod in ["rife4.6", "rife4.15", "rife4.15-lite"]:
                self.model = IFNet(
                    self.ensemble,
                    self.dynamicScale,
                    self.scale,
                    self.interpolateFactor,
                    self.staticStep,
                )
            else:
                self.model = IFNet(
                    self.ensemble,
                    self.dynamicScale,
                    self.scale,
                    self.interpolateFactor,
                )

        if checker.cuda_available and self.half:
            self.model.half()
        else:
            self.half = False
            self.model.float()

        self.model.load_state_dict(torch.load(modelPath, map_location=checker.device))
        self.model.eval().cuda() if checker.cuda_available else self.model.eval()
        self.model = self.model.to(checker.device)
        self.model = self.model.to(memory_format=torch.channels_last)

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

        if self.interpolateMethod in ["rife4.25", "rife4.25-heavy", "rife4.25-lite"]:
            ph = ((self.height - 1) // 128 + 1) * 128
            pw = ((self.width - 1) // 128 + 1) * 128
        else:
            ph = ((self.height - 1) // 64 + 1) * 64
            pw = ((self.width - 1) // 64 + 1) * 64
        self.padding = (0, pw - self.width, 0, ph - self.height)

        self.I0 = torch.zeros(
            1,
            3,
            self.height + self.padding[3],
            self.width + self.padding[1],
            dtype=self.dType,
            device=checker.device,
        ).to(memory_format=torch.channels_last)

        self.I1 = torch.zeros(
            1,
            3,
            self.height + self.padding[3],
            self.width + self.padding[1],
            dtype=self.dType,
            device=checker.device,
        ).to(memory_format=torch.channels_last)

        self.firstRun = True
        self.stream = torch.cuda.Stream()
        self.normStream = torch.cuda.Stream()

        self._timestep_buffer = torch.zeros(
            (1, 1, self.height + self.padding[3], self.width + self.padding[1]),
            dtype=self.dType,
            device=checker.device,
        )

    @torch.inference_mode()
    def cacheFrameReset(self, frame):
        self.processFrame(frame, "cache")
        self.processFrame(self.I0, "model")

    @torch.inference_mode()
    def processFrame(self, frame, toNorm):
        match toNorm:
            case "I0":
                with torch.cuda.stream(self.normStream):
                    frame = frame.to(
                        device=checker.device,
                        dtype=self.dType,
                        non_blocking=True,
                    )
                    frame = self.padFrame(frame)
                    self.I0.copy_(frame, non_blocking=True)
                # Sync only when data must be ready for inference
                self.normStream.synchronize()

            case "I1":
                with torch.cuda.stream(self.normStream):
                    frame = frame.to(
                        device=checker.device,
                        dtype=self.dType,
                        non_blocking=True,
                    )
                    frame = self.padFrame(frame)
                    self.I1.copy_(frame, non_blocking=True)
                # Sync only when data must be ready for inference
                self.normStream.synchronize()

            case "cache":
                with torch.cuda.stream(self.normStream):
                    self.I0.copy_(self.I1, non_blocking=True)
                    self.model.cache()
                self.normStream.synchronize()

            case "infer":
                with torch.cuda.stream(self.normStream):
                    if self.staticStep:
                        output = self.model(self.I0, self.I1, frame).clone()
                    else:
                        output = self.model(self.I0, self.I1, frame)[
                            :, :, : self.height, : self.width
                        ].clone()
                self.normStream.synchronize()
                return output

            case "model":
                with torch.cuda.stream(self.normStream):
                    self.model.cacheReset(frame)
                self.normStream.synchronize()

    @torch.inference_mode()
    def padFrame(self, frame):
        return (
            F.pad(frame, [0, self.padding[1], 0, self.padding[3]])
            if self.padding != (0, 0, 0, 0)
            else frame
        )

    @torch.inference_mode()
    def __call__(self, frame, interpQueue, framesToInsert: int = 2, timesteps=None):
        if self.firstRun:
            self.processFrame(frame, "I0")
            self.firstRun = False
            return
        self.processFrame(frame, "I1")

        for i in range(framesToInsert):
            if timesteps is not None and i < len(timesteps):
                t = timesteps[i]
            else:
                t = (i + 1) * 1 / (framesToInsert + 1)

            # Use pre-allocated buffer and fill in-place
            self._timestep_buffer.fill_(t)
            output = self.processFrame(self._timestep_buffer, "infer")
            interpQueue.put(output)

        self.processFrame(None, "cache")


class RifeTensorRT:
    def __init__(
        self,
        interpolateMethod: str = "rife4.25-tensorrt",
        interpolateFactor: int = 2,
        width: int = 0,
        height: int = 0,
        half: bool = True,
        ensemble: bool = False,
    ):
        """
        Interpolates frames using TensorRT

        Args:
            interpolateMethod (str, optional): Interpolation method. Defaults to "rife415".
            interpolateFactor (int, optional): Interpolation factor. Defaults to 2.
            width (int, optional): Width of the frame. Defaults to 0.
            height (int, optional): Height of the frame. Defaults to 0.
            half (bool, optional): Half resolution. Defaults to True.
            ensemble (bool, optional): Ensemble. Defaults to False.
        """
        import tensorrt as trt

        from .utils.trtHandler import (
            tensorRTEngineCreator,
            tensorRTEngineLoader,
            tensorRTEngineNameHandler,
        )

        self.tensorRTEngineCreator = tensorRTEngineCreator
        self.tensorRTEngineLoader = tensorRTEngineLoader
        self.tensorRTEngineNameHandler = tensorRTEngineNameHandler
        self.trt = trt

        self.interpolateMethod = interpolateMethod
        self.interpolateFactor = interpolateFactor
        self.width = width
        self.height = height
        self.half = half
        self.ensemble = ensemble
        self.model = None
        if self.width > 1920 and self.height > 1080:
            if self.half:
                logAndPrint(
                    "UHD and fp16 are not compatible with RIFE, defaulting to fp32",
                    "yellow",
                )
                logging.info(
                    "UHD and fp16 for rife are not compatible due to flickering issues, defaulting to fp32"
                )
                self.half = False
                # self.scale = 1.0
        else:
            pass
            # self.scale = 1.0
        self.scale = 1.0
        self.handleModel()

    def handleModel(self):
        if self.half:
            torch.set_default_dtype(torch.float16)
        self.filename = modelsMap(
            self.interpolateMethod.replace("-tensorrt", ""),
            modelType="pth",
            half=self.half,
            ensemble=self.ensemble,
        )

        folderName = self.interpolateMethod.replace("-tensorrt", "")
        if not os.path.exists(os.path.join(weightsDir, folderName, self.filename)):
            self.modelPath = downloadModels(
                model=self.interpolateMethod.replace("-tensorrt", ""),
                modelType="pth",
                half=self.half,
                ensemble=self.ensemble,
            )
        else:
            self.modelPath = os.path.join(weightsDir, folderName, self.filename)

        if self.interpolateMethod in [
            "rife_elexor-tensorrt",
            "rife4.25-tensorrt",
            "rife4.25-heavy-tensorrt",
        ]:
            channels = 4
            mul = 64
        elif self.interpolateMethod in [
            "rife4.22-lite-tensorrt",
        ]:
            channels = 4
            mul = 32
        elif self.interpolateMethod in [
            "rife4.25-lite-tensorrt",
        ]:
            channels = 4
            mul = 128
        else:
            channels = 8
            mul = 32

        self.dtype = torch.float16 if self.half else torch.float32
        tmp = max(mul, int(mul / 1.0))
        self.pw = math.ceil(self.width / tmp) * tmp
        self.ph = math.ceil(self.height / tmp) * tmp
        self.padding = (0, self.pw - self.width, 0, self.ph - self.height)

        IFNet, _Head = importRifeArch(self.interpolateMethod, "v3")

        enginePath = self.tensorRTEngineNameHandler(
            modelPath=self.modelPath,
            fp16=self.half,
            optInputShape=[1, 3, self.height, self.width],
            ensemble=self.ensemble,
            isRife=True,
        )

        self.model = IFNet(
            scale=self.scale,
            ensemble=self.ensemble,
            dtype=self.dtype,
            device=checker.device,
            width=self.width,
            height=self.height,
        )

        self.model.to(checker.device)
        if self.half:
            self.model.half()
        else:
            self.model.float()
        self.model.load_state_dict(torch.load(self.modelPath, map_location="cpu"))

        if _Head is True:
            self.norm = self.model.encode
        else:
            self.norm = None

        self.engine, self.context = self.tensorRTEngineLoader(enginePath)
        if (
            self.engine is None
            or self.context is None
            or not os.path.exists(enginePath)
        ):
            if ADOBE:
                progressState.update(
                    {"status": f"Exporting {self.interpolateMethod} to TensorRT."}
                )
            dummyInput1 = torch.zeros(
                1, 3, self.ph, self.pw, dtype=self.dtype, device=checker.device
            )
            dummyInput2 = torch.zeros(
                1, 3, self.ph, self.pw, dtype=self.dtype, device=checker.device
            )
            dummyInput3 = torch.full(
                (1, 1, self.ph, self.pw),
                0.5,
                dtype=self.dtype,
                device=checker.device,
            )

            if self.norm is not None:
                dummyInput4 = torch.zeros(
                    1,
                    channels,
                    self.ph,
                    self.pw,
                    dtype=self.dtype,
                    device=checker.device,
                )

            self.modelPath = self.modelPath.replace(".pth", ".onnx")

            inputList = [dummyInput1, dummyInput2, dummyInput3]
            inputNames = ["img0", "img1", "timestep"]
            outputNames = ["output"]
            dynamicAxes = {
                "img0": {2: "height", 3: "width"},
                "img1": {2: "height", 3: "width"},
                "timestep": {2: "height", 3: "width"},
                "output": {1: "height", 2: "width"},
            }

            if self.norm is not None:
                inputList.append(dummyInput4)
                inputNames.append("f0")
                outputNames.append("f1")
                dynamicAxes["f0"] = {2: "height", 3: "width"}

            torch.onnx.export(
                self.model,
                tuple(inputList),
                self.modelPath,
                input_names=inputNames,
                output_names=outputNames,
                dynamic_axes=dynamicAxes,
                opset_version=22,
                optimize=False,
                dynamo=False,
            )

            inputs = [
                [1, 3, self.ph, self.pw],
                [1, 3, self.ph, self.pw],
                [1, 1, self.ph, self.pw],
            ]

            if self.norm is not None:
                inputs.append([1, channels, self.ph, self.pw])

            if hasattr(self, "model") and self.model is not None:
                del self.model
                import gc

                del (
                    dummyInput1,
                    dummyInput2,
                    dummyInput3,
                )  # No need to keep these in memory

                if self.norm is not None:
                    del dummyInput4
                gc.collect()
                torch.cuda.empty_cache()

            inputsMin = inputsOpt = inputsMax = inputs

            logging.info("Loading engine failed, creating a new one")
            self.engine, self.context = self.tensorRTEngineCreator(
                modelPath=self.modelPath,
                enginePath=enginePath,
                fp16=self.half,
                inputsMin=inputsMin,
                inputsOpt=inputsOpt,
                inputsMax=inputsMax,
                inputName=inputNames,
                isMultiInput=True,
            )

            try:
                os.remove(self.modelPath)

            except Exception as e:
                logging.error(f"Error removing onnx model: {e}")

        self.dtype = torch.float16 if self.half else torch.float32
        self.stream = torch.cuda.Stream()
        self.I0 = torch.zeros(
            1,
            3,
            self.ph,
            self.pw,
            dtype=self.dtype,
            device=checker.device,
        )

        self.I1 = torch.zeros(
            1,
            3,
            self.ph,
            self.pw,
            dtype=self.dtype,
            device=checker.device,
        )

        if self.norm is not None:
            self.f0 = torch.zeros(
                1,
                channels,
                self.ph,
                self.pw,
                dtype=self.dtype,
                device=checker.device,
            )

            self.f1 = torch.zeros(
                1,
                channels,
                self.ph,
                self.pw,
                dtype=self.dtype,
                device=checker.device,
            )

        self.dummyTimeStep = torch.full(
            (1, 1, self.ph, self.pw),
            0.5,
            dtype=self.dtype,
            device=checker.device,
        )

        self.dummyOutput = torch.zeros(
            (1, 3, self.height, self.width),
            device=checker.device,
            dtype=self.dtype,
        )

        self.tensors = [
            self.I0,
            self.I1,
            self.dummyTimeStep,
        ]

        if self.norm is not None:
            self.tensors.extend([self.f0])

        self.tensors.extend([self.dummyOutput])

        if self.norm is not None:
            self.tensors.extend([self.f1])

        self.bindings = [tensor.data_ptr() for tensor in self.tensors]

        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            self.context.set_tensor_address(tensor_name, self.bindings[i])

            if self.engine.get_tensor_mode(tensor_name) == self.trt.TensorIOMode.INPUT:
                self.context.set_input_shape(tensor_name, self.tensors[i].shape)

        self.firstRun = True
        self.normStream = torch.cuda.Stream()
        self.outputStream = torch.cuda.Stream()

        self.cudaGraph = torch.cuda.CUDAGraph()
        self.initTorchCudaGraph()

    @torch.inference_mode()
    def initTorchCudaGraph(self):
        with torch.cuda.graph(self.cudaGraph, stream=self.stream):
            self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()

    @torch.inference_mode()
    def processFrame(self, frame, name=None):
        with torch.cuda.stream(self.normStream):
            match name:
                case "I0":
                    self.I0.copy_(
                        F.pad(
                            frame.to(dtype=self.dtype),
                            self.padding,
                        ),
                        non_blocking=True,
                    )

                case "I1":
                    self.I1.copy_(
                        F.pad(
                            frame.to(dtype=self.dtype),
                            self.padding,
                        ),
                        non_blocking=True,
                    )

                case "f0":
                    self.f0.copy_(
                        self.norm(
                            F.pad(
                                frame.to(dtype=self.dtype),
                                self.padding,
                            )
                        ),
                        non_blocking=True,
                    )

                case "f0-copy":
                    self.f0.copy_(self.f1, non_blocking=True)

                case "cache":
                    self.I0.copy_(self.I1, non_blocking=True)

                case "timestep":
                    self.dummyTimeStep.copy_(frame, non_blocking=True)

        self.normStream.synchronize()

    @torch.inference_mode()
    def cacheFrameReset(self, frame):
        self.processFrame(frame, "I0")

    @torch.inference_mode()
    def __call__(self, frame, interpQueue, framesToInsert=1, timesteps=None):
        if self.firstRun:
            if self.norm is not None:
                self.processFrame(frame, "f0")

            self.processFrame(frame, "I0")

            self.firstRun = False
            return

        self.processFrame(frame, "I1")
        for i in range(framesToInsert):
            if timesteps is not None and i < len(timesteps):
                t = timesteps[i]
            else:
                t = (i + 1) * 1 / (framesToInsert + 1)

            self.dummyTimeStep.fill_(t)

            self.processFrame(self.dummyTimeStep, "timestep")
            with torch.cuda.stream(self.stream):
                self.cudaGraph.replay()
            self.stream.synchronize()
            with torch.cuda.stream(self.outputStream):
                output = self.dummyOutput.clone()
            self.outputStream.synchronize()
            interpQueue.put(output)

        self.processFrame(None, "cache")
        if self.norm is not None:
            self.processFrame(None, "f0-copy")
