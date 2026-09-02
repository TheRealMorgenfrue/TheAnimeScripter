from typing import Literal, Self, override

from src.module.utils.hardware_checkers.checker_base import (
    HardwareCheckerBase,
    HardwareDevice,
)


class AmdChecker(HardwareCheckerBase):
    _instance = None

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._created = False
        return cls._instance

    def __init__(self):
        if not self._created:
            super().__init__()
            self._rocm_available = False  # TODO: Implement
            # try:
            #     self._cuda_available = torch.cuda.is_available()
            #     if self._cuda_available:
            #         test = torch.zeros(1, device="cuda")
            #         _ = test + 1
            #         del test
            # except Exception:
            #     self._cuda_available = False
            #     self.logger.warning(f"CUDA is not available:\n{traceback.format_exc()}")

            # if self._cuda_available:
            #     self.enable_cuda_optimizations()

            self._created = True

    @override
    def detect_device_architectures(self) -> list[HardwareDevice]:
        return []

    @override
    def get_execution_providers(
        self, requested_devices: set[int] | Literal["all"]
    ) -> dict[int, list[str]]:
        return {}
