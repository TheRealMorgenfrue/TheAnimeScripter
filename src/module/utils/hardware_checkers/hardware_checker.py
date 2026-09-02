from typing import Literal, Self, override

from src.module.utils.hardware_checkers.amd_checker import AmdChecker
from src.module.utils.hardware_checkers.apple_checker import AppleChecker
from src.module.utils.hardware_checkers.checker_base import (
    HardwareCheckerBase,
    HardwareDevice,
)
from src.module.utils.hardware_checkers.intel_checker import IntelChecker
from src.module.utils.hardware_checkers.nvidia_checker import NvidiaChecker


class HardwareChecker(HardwareCheckerBase):
    _instance = None

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._created = False
        return cls._instance

    def __init__(self) -> None:
        if not self._created:
            super().__init__()
            self.nvidia = NvidiaChecker()
            self.amd = AmdChecker()
            self.intel = IntelChecker()
            self.apple = AppleChecker()

            self.checkers = [self.nvidia, self.amd, self.intel, self.apple]
            self._device_archs = None
            self._name_to_id = None
            self._id_to_name = None

            self._created = True

    @property
    def cuda_available(self):
        """Whether CUDA is available.

        NVIDIA hardware only.
        """
        return self.nvidia._cuda_available

    @property
    def mps_available(self):
        """Whether Metal Performance Shaders (MPS) are available.

        Apple hardware only.
        """
        return self.apple._mps_available

    @property
    def openvino_available(self):
        """Whether OpenVINO is available.

        Intel hardware only.
        """
        return self.intel._openvino_available

    @property
    def rocm_available(self):
        """Whether ROCm is available.

        AMD hardware only.
        """
        return self.amd._rocm_available

    def get_device_id(self, device_name: str) -> int | None:
        """Returns the device_id of the device_name or None if not found."""
        if self._name_to_id is None:
            self.detect_device_architectures()
        self._name_to_id.get(device_name)  # type: ignore

    def get_device_name(self, device_id: int) -> str | None:
        """Returns the device_name of the device_id or None if not found."""
        if self._id_to_name is None:
            self.detect_device_architectures()
        self._id_to_name.get(device_id)  # type: ignore

    @override
    def detect_device_architectures(self) -> list[HardwareDevice]:
        if self._device_archs is None:
            devices = []
            for checker in self.checkers:
                devices.extend(checker.detect_device_architectures())
            self._device_archs = devices
            self._name_to_id = {
                device.name: i for i, device in enumerate(self._device_archs)
            }
            self._id_to_name = {v: k for k, v in self._name_to_id.items()}
        return self._device_archs

    @override
    def get_execution_providers(
        self, requested_devices: set[int] | Literal["all"]
    ) -> dict[int, list[str]]:
        providers = {}
        for checker in self.checkers:
            providers.update(checker.get_execution_providers(requested_devices))
        return providers
