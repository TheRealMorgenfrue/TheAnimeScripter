from typing import Self, override

from applib import (
    BaseTemplate,
    ComboBoxOption,
    GenericConverter,
    GUIMessage,
    Option,
    UITypes,
)

from src.module.config.tas_args import TASArgs


# Defined by: https://onnxruntime.ai/docs/api/python/api_summary.html#runoptions
class RunOptionsTemplate(BaseTemplate):
    """ONNX Run Options"""

    _instance = None

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._created = False
        return cls._instance

    def __init__(self) -> None:
        if not self._created:
            super().__init__(
                name=TASArgs.onnx_run_options_template_name,
                template=self._create_template(),
                icons=None,
            )
            self._created = True

    @override
    def _create_template(self) -> dict:
        return {
            "disable_synchronize_execution_providers": Option(
                default="0",
                converter=GenericConverter(["0", "1"], [False, True]),
                ui_type=UITypes.SWITCH,
                ui_info=GUIMessage(
                    "Perform asynchronous copies while running inference",
                    "Increases performance by hiding up and downloads for inputs behind inference",
                ),
            ),
            "log_severity_level": ComboBoxOption(
                default="2",
                values={
                    "Verbose": "0",
                    "Info": "1",
                    "Warning": "2",
                    "Error": "3",
                    "Fatal": "4",
                },
                ui_info=GUIMessage(
                    "Log level at runtime",
                ),
            ),
        }
