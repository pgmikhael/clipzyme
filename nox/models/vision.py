import torchvision.models as models
import torch.nn as nn
import torch
from nox.utils.registry import register_object
from nox.models.abstract import AbstractModel

# from efficientnet_pytorch import EfficientNet
import math


@register_object("resnet18", "model")
class Resnet18(AbstractModel):
    def __init__(self, args):
        super(Resnet18, self).__init__()
        model = models.resnet18(pretrained=args.trained_on_imagenet)
        if args.num_chan != 3:
            w = model._modules["conv1"].weight.repeat(
                1, math.ceil(args.num_chan / 3), 1, 1
            )[:, : args.num_chan]
            model._modules["conv1"].weight = torch.nn.Parameter(w)

        modules = list(model.children())[:-1]
        self._model = nn.Sequential(*modules)

    def forward(self, batch=None):
        return {"hidden": self._model(batch["x"])[:, :, 0, 0]}

    @property
    def hidden_dim(self):
        return 512

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--trained_on_imagenet",
            action="store_true",
            default=False,
            help="Use weights pretrained on image net",
        )
