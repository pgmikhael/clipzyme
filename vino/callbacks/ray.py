import os
from vino.utils.registry import register_object
from vino.utils.classes import Vino
from ray_lightning.tune import TuneReportCallback

# TODO: add args for various callbacks -- currently hardcoded


@register_object("tune_report", "callback")
class TuneReport(TuneReportCallback, Vino):
    def __init__(self, args) -> None:
        super().__init__(
            metrics={args.monitor: args.monitor},
            on="validation_end",
        )
