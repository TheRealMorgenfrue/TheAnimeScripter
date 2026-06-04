from applib import CoreHomeInterface

from src.module.config.tas_config import TASConfig


class TASHomeInterface(CoreHomeInterface):
    def __init__(self, parent=None):
        super().__init__(TASConfig(), parent)
