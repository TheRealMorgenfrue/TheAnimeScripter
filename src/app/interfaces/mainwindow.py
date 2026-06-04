from applib import CoreMainWindow
from qfluentwidgets import FluentIcon as FIF

from src.app.interfaces.home import TASHomeInterface
from src.app.interfaces.settings import TASSettingsInterface
from src.module.config.tas_args import TASArgs
from src.module.config.tas_config import TASConfig


class TASMainWindow(CoreMainWindow):
    def __init__(self):
        super().__init__(
            MainArgs=TASArgs,
            MainConfig=TASConfig,
            subinterfaces=[
                (TASHomeInterface, FIF.HOME, "Home"),
            ],
            settings_interface=(TASSettingsInterface, FIF.SETTING, "Settings"),
        )
