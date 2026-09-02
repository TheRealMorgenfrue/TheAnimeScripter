import traceback

import nelux
from applib import LoggingManager


def set_nelux_log_level(level: str | int | nelux.LogLevel):
    logger = LoggingManager()

    _level = None
    if isinstance(level, nelux.LogLevel):
        _level = level
    elif isinstance(level, int):
        for enum in nelux.LogLevel._member_map_.values():
            if enum.value == level:
                _level = enum
                break
    elif isinstance(level, str):
        for name in nelux.LogLevel._member_names_:
            if level.lower() == name.lower():
                _level = nelux.LogLevel._member_map_[name]
                break

    if _level is None:
        logger.error(
            f"Invalid NeLux loglevel '{level}'. Expected a value in Enum '{nelux.LogLevel.__name__}'",
            gui=True,
        )
    else:
        try:
            nelux.set_log_level(_level)  # type: ignore
        except Exception:
            logger.error(
                f"Failed to enable NeLux logging\n{traceback.format_exc()}", gui=True
            )
