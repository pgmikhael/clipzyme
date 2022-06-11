import torch
from torch import optim
from vino.utils.registry import register_object
from vino.utils.classes import Vino


@register_object("sgd", "optimizer")
class SGD(optim.SGD, Vino):
    """
    https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
    """

    def __init__(self, params, args):
        super().__init__(
            params=params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
        )


@register_object("adam", "optimizer")
class Adam(optim.Adam, Vino):
    """
    https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
    """

    def __init__(self, params, args):
        super().__init__(params=params, lr=args.lr, weight_decay=args.weight_decay)
