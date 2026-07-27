import os
from pathlib import Path
from typing import Any

from module.config.olive_config import OliveConfig
from module.models.tensor_proto_map import TensorProtoMap
from module.models.vfi.vfi_tensors_base import VFIModelTensors
from src.module.config.tas_config import TASConfig
from src.module.models.model_base import ModelBase
from src.module.models.vfi.vfi_base import VFIModelBase


class ModelHandler:
    def __init__(self) -> None:
        olive_config = OliveConfig()
        self.tas_config = TASConfig()
        self.model_extensions = ["pth", "pt", "onnx", "safetensors"]
        self.model_params = {
            "sr": {},
            "vfi": {
                "width": 1920,
                "height": 1080,
                "padded_width": 1920,
                "padded_height": 1080,
                "multiplier": 32,
                "channels": 8,
                "dtype": TensorProtoMap().get_torch(
                    olive_config.get_value("dtype", path="onnx_conversion")
                ),
                "device": olive_config.get_value("device", path="onnx_conversion"),
            },
        }

    def initialize_models(self, width: int, height: int) -> list[ModelBase]:
        processes = []

        if self.tas_config["vfi"]:
            model_path = self._get_model_path(
                self.tas_config["vfi_model"], self.tas_config["model_dir"]
            )
            params = self.get_model_parameters(f"{model_path}")["vfi"]
            vfi_tensors = VFIModelTensors(
                width=width,
                height=height,
                multiplier=params["multiplier"],
                channels=params["channels"],
            )

            self.model_params["vfi"]["width"] = width
            self.model_params["vfi"]["heigth"] = height
            self.model_params["vfi"]["padded_width"] = vfi_tensors.padded_width
            self.model_params["vfi"]["padded_height"] = vfi_tensors.padded_height

            vfi = VFIModelBase(
                model_path=model_path,
                vfi_factor=self.tas_config["vfi_factor"],
                tensors=vfi_tensors,
                dtype=self.model_params["vfi"]["dtype"],
            )
            processes.append(vfi)
        return processes

    def _get_model_path(self, model_name: str, model_dir: str) -> Path:
        for ext in self.model_extensions:
            path = Path(model_dir, f"{model_name}.{ext}")
            if path.exists():
                return path
        else:
            raise RuntimeError(
                f"Failed to find model '{model_name}' in folder '{model_dir}'"
            )

    def _get_model_type(self, model_path: str) -> str:
        path = Path(model_path)
        model_types = ["vfi", "sr"]
        for model_type in model_types:
            if path.match(model_type):
                return model_type
        raise RuntimeError(
            f"Model path {model_path!r} did not match any model type in {model_types}"
        )

    def get_model_parameters(
        self,
        model_path: str,
    ) -> dict[str, Any]:
        model_name = os.path.splitext(os.path.split(model_path)[1])[0]
        model_type = self._get_model_type(model_path)
        match model_type:
            case "sr":
                pass
            case "vfi":
                match model_name:
                    case "rife_elexor":
                        self.model_params["vfi"]["multiplier"] = 64
                        self.model_params["vfi"]["channels"] = 4
                    case _:
                        self.model_params["vfi"]["multiplier"] = 32
                        self.model_params["vfi"]["channels"] = 8

        return self.model_params
