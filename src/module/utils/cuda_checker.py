import glob
import os
import platform
import shutil
import subprocess
import traceback
from typing import Self

import torch
from applib import LoggingManager


class CudaChecker:
    """
    A dumb class to check if CUDA is available and to get the device name.
    Just to avoid writing the same code over and over again.

    Note: This class checks if CUDA is available in PyTorch, but does not
    validate if the GPU architecture is compatible with modern CUDA kernels.
    Use self.detect_gpu_architecture() to check for Pascal or older GPUs.
    """

    _instance = None

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._created = False
        return cls._instance

    def __init__(self):
        if not self._created:
            self.logger = LoggingManager()
            try:
                self._cuda_available = torch.cuda.is_available()
                if self._cuda_available:
                    test = torch.zeros(1, device="cuda")
                    _ = test + 1
                    del test
            except Exception as e:
                self._cuda_available = False
                self.logger.warning(f"CUDA is not available: {e}")

            if self._cuda_available:
                self.enable_cuda_optimizations()

            self._created = True

    @property
    def cuda_available(self):
        return self._cuda_available

    @property
    def device(self):
        return torch.device("cuda" if self.cuda_available else "cpu")

    @property
    def device_name(self):
        if not self.cuda_available:
            return "cpu"
        try:
            return torch.cuda.get_device_name(0)
        except (RuntimeError, AssertionError):
            self.logger.warning(
                f"Could not get CUDA device name:\n{traceback.format_exc()}"
            )
            return "cpu"

    @property
    def device_count(self):
        """Get the number of available CUDA devices."""
        return torch.cuda.device_count() if self.cuda_available else 0

    @property
    def all_device_names(self):
        """Get the names of all available CUDA devices."""
        if not self.cuda_available:
            return ["cpu"]

        # A for-loop is used to prevent errors from aborting the entire process
        # Thus, we get as many device names as possible.
        device_names: list[str] = []
        try:
            for i in range(self.device_count):
                device_names.append(torch.cuda.get_device_name(i))  # noqa: PERF401
        except (RuntimeError, AssertionError):
            self.logger.warning(
                f"Could not get all CUDA device names:\n{traceback.format_exc()}"
            )
        return device_names if device_names else "cpu"

    def enable_cuda_optimizations(self):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True

    def disable_cuda_optimizations(self):
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

    def run_nvsmi_check(self, path: str) -> bool:
        cmd = [path, "-L"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode == 0 and result.stdout:
                gpu_lines = [
                    line.strip()
                    for line in result.stdout.strip().split("\n")
                    if line.strip()
                ]
                if gpu_lines:
                    self.logger.info(f"NVIDIA GPUs detected: {', '.join(gpu_lines)}")
                    return True
        except (subprocess.SubprocessError, FileNotFoundError):
            self.logger.warning(
                f"nvidia-smi not found or failed to run\n{traceback.format_exc()}"
            )
            self.logger.debug(f"nvidia-smi command used: {cmd}")
        return False

    def check_windows_adapters(self) -> bool:
        cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0 and result.stdout:
                adapters = result.stdout.lower()
                return "nvidia" in adapters
        except (subprocess.SubprocessError, FileNotFoundError):
            self.logger.error(
                f"Failed to check Windows adapters\n{traceback.format_exc()}"
            )
            self.logger.debug(f"Windows adapter command used: {cmd}")
        return False

    def check_linux_pci(self) -> bool:
        vendor_paths = glob.glob("/sys/bus/pci/devices/*/vendor")
        for vendor_path in vendor_paths:
            try:
                with open(vendor_path, encoding="utf-8") as handle:
                    if handle.read().strip().lower() == "0x10de":
                        return True
            except OSError:
                continue
        try:
            result = subprocess.run(
                ["lspci"], capture_output=True, text=True, check=False
            )
            if result.returncode == 0 and "nvidia" in result.stdout.lower():
                return True
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        return False

    def parse_compute_capability(self, value: str) -> tuple[int | None, int | None]:
        try:
            parts = value.split(".")
            major = int(parts[0])
            minor = int(parts[1]) if len(parts) > 1 else 0
            return major, minor
        except (ValueError, AttributeError):
            return None, None

    def is_turing_or_newer(self, major: int, minor: int) -> bool:
        if major > 7:
            return True
        if major == 7 and minor >= 5:
            return True
        return False

    def detect_nvidia_gpu(self) -> bool:
        for path in self.get_nvsmi_paths():
            if os.path.exists(path) and self.run_nvsmi_check(path):
                return True
        system_name = platform.system().lower()
        if system_name == "windows" and self.check_windows_adapters():
            self.logger.info("NVIDIA GPU detected via WMI")
            return True
        if system_name == "linux" and self.check_linux_pci():
            self.logger.info("NVIDIA GPU detected via PCI scan")
            return True
        self.logger.info("No NVIDIA GPU detected")
        return False

    def detect_gpu_architecture(self) -> tuple[bool, str | None, str | None]:
        """
        Returns:
            tuple: (is_modern: bool, gpu_name: str | None, compute_capability: str | None)
                is_modern is True for Turing (compute 7.0) and newer architectures
        """
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode == 0 and result.stdout:
                lines = result.stdout.strip().split("\n")
                if lines:
                    parts = lines[0].split(",")
                    if len(parts) >= 2:
                        gpu_name = parts[0].strip()
                        compute_capability = parts[1].strip()
                        major, minor = self.parse_compute_capability(compute_capability)
                        # TODO: Make more robust GPU checking. Also check for stuff like TF32 / BF16 etc.
                        if major is not None and minor is not None:
                            is_modern = self.is_turing_or_newer(major, minor)
                            if not is_modern:
                                self.logger.debug(
                                    f"GPU {gpu_name} has compute capability {compute_capability} (Pascal or older). DirectML backend recommended."
                                )
                            else:
                                self.logger.debug(
                                    f"GPU {gpu_name} has compute capability {compute_capability} - modern CUDA support available"
                                )
                        else:
                            is_modern = False
                        return is_modern, gpu_name, compute_capability

            result = subprocess.run(
                ["nvidia-smi", "-L"], capture_output=True, text=True, check=False
            )

            if result.returncode == 0 and result.stdout:
                gpu_name = result.stdout.strip().split("\n")[0]
                if ":" in gpu_name:
                    gpu_name = gpu_name.split(":")[1].strip().split("(")[0].strip()

                oldArchitectures = [
                    "GTX 9",
                    "GTX 10",
                    "GT 7",
                    "GT 8",
                    "GT 9",
                    "Quadro K",
                    "Quadro M",
                    "Quadro P",
                    "Tesla K",
                    "Tesla M",
                    "Tesla P",
                ]

                isOld = any(arch in gpu_name for arch in oldArchitectures)

                if isOld:
                    self.logger.info(
                        f"GPU {gpu_name} appears to be Pascal generation or older. DirectML backend recommended for compatibility."
                    )
                    return False, gpu_name, None

                return True, gpu_name, None

        except (subprocess.SubprocessError, FileNotFoundError):
            self.logger.error(
                f"Could not detect GPU architecture:\n{traceback.format_exc()}"
            )

        return False, None, None
