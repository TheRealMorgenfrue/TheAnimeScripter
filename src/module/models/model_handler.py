from pathlib import Path

from src.module.config.tas_config import TASConfig
from src.module.models.model_base import ModelBase
from src.module.models.vfi.vfi_base import VFIModelBase


class ModelHandler:
    def __init__(self) -> None:
        self.model_extensions = ["pth", "pt", "onnx", "safetensors"]

    def initialize_models(self, width: int, height: int) -> list[ModelBase]:
        config = TASConfig()
        processes = []

        if config["vfi"]:
            vfi = VFIModelBase(
                model_path=self._get_model_path(
                    config["vfi_model"], config["model_dir"]
                ),
                width=width,
                height=height,
                vfi_factor=config["vfi_factor"],
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
