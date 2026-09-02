import os
import shutil
import subprocess
import traceback
from typing import Literal, Self, override

import torch
from applib import iter_to_str

from src.module.utils.hardware_checkers.checker_base import (
    HardwareCheckerBase,
    HardwareDevice,
)


class NvidiaChecker(HardwareCheckerBase):
    class ComputeCap:
        def __init__(self, name: str, cap: tuple[int, int]) -> None:
            self.name = name
            self.value = cap

        def __repr__(self) -> str:
            return (
                f"{iter_to_str(self.value, separator='.')} ({self.name.capitalize()})"
            )

    _instance = None

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._created = False
        return cls._instance

    def __init__(self) -> None:
        if not self._created:
            super().__init__()
            try:
                self._cuda_available = torch.cuda.is_available()
                if self._cuda_available:
                    test = torch.zeros(1, device="cuda")
                    _ = test + 1
                    del test
            except Exception:
                self._cuda_available = False
                self.logger.warning(
                    f"CUDA is not available:\n{traceback.format_exc()}", gui=True
                )

            if self._cuda_available:
                self.enable_cuda_optimizations()

            self._created = True

    def enable_cuda_optimizations(self) -> None:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True

    def disable_cuda_optimizations(self) -> None:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

    def get_nvsmi_paths(self) -> set[str]:
        paths = set()

        path = shutil.which("nvidia-smi")
        systemRoot = os.environ.get("SystemRoot", r"C:\\Windows")
        programFiles = os.environ.get("ProgramFiles", r"C:\\Program Files")
        programFilesX86 = os.environ.get(
            "ProgramFiles(x86)", r"C:\\Program Files (x86)"
        )

        if path:
            paths.add(path)

        paths.add(os.path.join(systemRoot, "System32", "nvidia-smi.exe"))
        paths.add(
            os.path.join(programFiles, "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe")
        )
        paths.add(
            os.path.join(
                programFilesX86, "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe"
            )
        )
        return paths

    def parse_compute_capability(self, value: str) -> tuple[int | None, int | None]:
        try:
            parts = value.split(".")
            major = int(parts[0])
            minor = int(parts[1]) if len(parts) > 1 else 0
            return major, minor
        except (ValueError, AttributeError):
            return None, None

    @override
    def detect_device_architectures(self) -> list[HardwareDevice]:
        gpu_architectures: dict[int, HardwareDevice] = {}
        smi_path = "nvidia-smi"
        for path in self.get_nvsmi_paths():
            if os.path.exists(path):
                smi_path = path

        try:
            result = subprocess.run(
                [
                    smi_path,
                    "--query-gpu=name,compute_cap,index",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            lines = result.stdout.strip().split("\n")
            for line in lines:
                parts = line.split(",")
                try:
                    gpu_name = parts[0].strip()
                    compute_capability = parts[1].strip()
                    index = parts[2].strip()
                except IndexError:
                    self.logger.warning(
                        "nvidia-smi produced unexpected output", gui=True
                    )
                    self.logger.debug(f"nvidia-smi output: {line}", gui=True)
                    continue

                major, minor = self.parse_compute_capability(compute_capability)
                supported_cap = None
                match (major, minor):
                    case (3, _):
                        supported_cap = self.ComputeCap("Kepler", (major, minor or 0))
                    case (5, _):
                        supported_cap = self.ComputeCap("Maxwell", (major, minor or 0))
                    case (6, _):
                        supported_cap = self.ComputeCap("Pascal", (major, minor or 0))
                    case (7, 0):
                        supported_cap = self.ComputeCap("Volta", (major, minor))
                    case (7, 5):
                        supported_cap = self.ComputeCap("Turing", (major, minor))
                    case (8, _) if minor != 9:
                        supported_cap = self.ComputeCap("Ampere", (major, minor or 0))
                    case (8, 9):
                        supported_cap = self.ComputeCap("Ada", (major, minor))
                    case (9, 0):
                        supported_cap = self.ComputeCap("Hopper", (major, minor))
                    case (_, _) if major and 10 <= major <= 12:
                        supported_cap = self.ComputeCap(
                            "Blackwell", (major, minor or 0)
                        )

                if supported_cap is None:
                    self.logger.warning(
                        f'GPU {index} "{gpu_name}" has unsupported compute capability {compute_capability}. '
                        f"Minimum supported is 3.0 (Kepler)",
                        gui=True,
                    )
                gpu_architectures[int(index)] = HardwareDevice(
                    name=gpu_name, metadata=supported_cap
                )
        except (subprocess.SubprocessError, FileNotFoundError):
            self.logger.error(
                f"Could not detect GPU architecture:\n{traceback.format_exc()}",
                gui=True,
            )
        return [gpu_architectures[i] for i in sorted(gpu_architectures.keys())]

    @override
    def get_execution_providers(
        self, requested_devices: set[int] | Literal["all"]
    ) -> dict[int, list[str]]:
        providers = {}
        avail_devices = self.detect_device_architectures()
        for i, device in enumerate(avail_devices):
            if requested_devices == "all" or i in requested_devices:
                cap: NvidiaChecker.ComputeCap = device.metadata
                if cap:
                    ep = []
                    if cap.value[0] >= 7:
                        ep.append("TensorrtExecutionProvider")
                    ep.append("CUDAExecutionProvider")
                    providers[i] = ep
        return providers


if __name__ == "__main__":
    a = NvidiaChecker().detect_device_architectures()
    print([(i.name, i.metadata) for i in a])
    b = NvidiaChecker().get_execution_providers("all")
    print(b)
