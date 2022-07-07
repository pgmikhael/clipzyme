from nox.utils.registry import register_object
import pytorch_lightning as pl
import os
from nox.utils.classes import Nox


@register_object("comet", "logger")
class COMET(pl.loggers.CometLogger, Nox):
    def __init__(self, args) -> None:
        super().__init__(
            api_key=os.environ.get("COMET_API_KEY"),
            project_name=args.project_name,
            experiment_name=args.experiment_name,
            workspace=args.workspace,
            log_env_details=True,
            log_env_cpu=True,
        )

    def setup(self, **kwargs):
        self.experiment.set_model_graph(kwargs["model"])
        self.experiment.add_tags(kwargs["args"].logger_tags)
        self.experiment.log_parameters(kwargs["args"])

    def log_image(self, image, name):
        self.experiment.log_image(image, name)
