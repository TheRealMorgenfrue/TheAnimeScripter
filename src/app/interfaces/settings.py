from applib import (
    CardGenerator,
    CoreSettingsInterface,
    CoreSettingsSubInterface,
    PivotCardStack,
)
from PyQt6.QtWidgets import QWidget
from qfluentwidgets import FluentIcon as FIF

from src.module.config.tas_args import TASArgs
from src.module.config.tas_config import TASConfig
from src.module.config.templates.tas_template import TASTemplate


class TASSettingsInterface(CoreSettingsInterface):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent=parent)
        self.addSubInterface(
            icon=FIF.AIRPLANE,
            title=TASArgs.name,
            widget=CoreSettingsSubInterface(
                config=TASConfig(),
                template=TASTemplate(),
                Generator=CardGenerator,
                CardStack=PivotCardStack,
            ),
        )
