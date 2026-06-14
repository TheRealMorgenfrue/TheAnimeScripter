from enum import Enum


class NeluxLogLevel(Enum):
    trace = 0
    debug = 1
    info = 2
    warn = 3
    error = 4
    critical = 5
    off = 6
