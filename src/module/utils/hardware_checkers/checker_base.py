from abc import abstractmethod
from typing import Any, Literal

from applib import LoggingManager


class HardwareDevice:
    def __init__(self, name: str, metadata: Any) -> None:
        self.name = name
        self.metadata = metadata


class HardwareCheckerBase:
    """Base class for hardware checkers.

    A hardware checker figures out what type of hardware is available.
    """

    def __init__(self) -> None:
        self.logger = LoggingManager()

    @abstractmethod
    def detect_device_architectures(
        self,
    ) -> list[HardwareDevice]:
        """Returns a list of all detected devices and their architecture.

        This includes, but is not limited to: CPUs, GPUs and NPUs.
        """
        ...

    @abstractmethod
    def get_execution_providers(
        self, requested_devices: set[int] | Literal["all"]
    ) -> dict[int, list[str]]:
        """Returns the execution providers compatible with each device in `requested_devices`.

        Parameters
        ----------
        requested_devices : set[int] | Literal[&quot;all&quot;]
            The IDs of the devices to consider when searching for execution providers.

            `all` considers all available devices.

        Returns
        -------
        dict[int, list[str]]
            A dict containing the device ID and a list of execution providers for each device.

            For instance:
            ```
            {
                0: ["TensorrtExecutionProvider", "CUDAExecutionProvider"],
                1: ["CUDAExecutionProvider"],
            }
            ```
        """
        ...
