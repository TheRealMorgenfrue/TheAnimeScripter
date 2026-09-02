from applib import LoggingManager

from src.module.utils.hardware_checkers.hardware_checker import HardwareChecker


def validate_devices(devices: list[str]) -> list[str]:
    """Ensure all selected devices exists on the system.

    Parameters
    ----------
    devices : list[str]
        The list of selected devices.

    Returns
    -------
    list[str]
        The list of devices which exist on the system.
    """
    all_devices = {
        device.name for device in HardwareChecker().detect_device_architectures()
    }
    selected = set(devices)
    valid_selected = {
        device_name for device_name in devices if device_name in all_devices
    }
    len_selected = len(devices)
    len_valid = len(valid_selected)
    if len_selected != len_valid:
        LoggingManager().warning(
            f"Failed to find {len_selected - len_valid} devices:\n\t{sorted(selected.difference(valid_selected))}",
            gui=True,
        )
        return sorted(valid_selected)
    else:
        return devices
