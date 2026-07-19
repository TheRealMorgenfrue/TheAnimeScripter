from pathlib import Path
from typing import override

import torch
from torch import Tensor

from src.module.config.io_binding_config import IOBindingConfig
from src.module.models.model_base import ModelBase


class VFIModelBase(ModelBase):
    """Base class for VFI models"""

    def __init__(self, model_path: Path) -> None:
        super().__init__(model_path, {}, {})

    def cache_frame_reset(self, frame: Tensor) -> None:
        """
        Reset the temporal state with a new frame.
        Called on scene changes.
        """
        self.process_frame(frame, "I0")

    def process_frame(self, frame: Tensor, name: str) -> None: ...

    @override
    def __call__(self, frame: Tensor, **kwargs) -> None:
        pass
