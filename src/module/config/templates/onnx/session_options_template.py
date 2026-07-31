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


# Defined by: https://onnxruntime.ai/docs/api/python/api_summary.html#sessionoptions
class SessionOptionsTemplate(BaseTemplate):
    """ONNX Session Options"""

    _instance = None

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._created = False
        return cls._instance

    def __init__(self) -> None:
        if not self._created:
            super().__init__(
                name=TASArgs.onnx_session_options_template_name,
                template=self._create_template(),
                icons=None,
            )
            self._created = True

    @override
    def _create_template(self) -> dict:
        return {
            "enable_cpu_mem_arena": Option(
                default="1",
                converter=GenericConverter(["0", "1"], [False, True]),
                ui_type=UITypes.SWITCH,
                ui_info=GUIMessage("Enables the memory arena on CPU"),
            ),
            "enable_profiling": Option(
                default="0",
                converter=GenericConverter(["0", "1"], [False, True]),
                ui_type=UITypes.SWITCH,
                ui_info=GUIMessage("Enable profiling for this session"),
            ),
            "session_log_severity_level": ComboBoxOption(
                default="2",
                values={
                    "Verbose": "0",
                    "Info": "1",
                    "Warning": "2",
                    "Error": "3",
                    "Fatal": "4",
                },
                ui_info=GUIMessage(
                    "Log level of session load, initialization, etc.",
                    "Does not apply to runtime logging.",
                ),
            ),
            "graph_optimization_level": ComboBoxOption(
                default="Full",
                values=[
                    "Disabled",
                    "Light",
                    "Extended",
                    "Full",
                ],
                ui_info=GUIMessage("Graph optimizations to improve performance"),
            ),
        }
