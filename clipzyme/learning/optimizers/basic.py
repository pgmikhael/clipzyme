import torch
from torch import optim
from clipzyme.utils.registry import register_object
from clipzyme.utils.classes import Nox


@register_object("sgd", "optimizer")
class SGD(optim.SGD, Nox):
    """
    https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
    """

    def __init__(self, params, args):
        super().__init__(
            params=params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )


@register_object("adam", "optimizer")
class Adam(optim.Adam, Nox):
    """
    https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
    """

    def __init__(self, params, args):
        adam_betas = tuple(args.adam_betas)
        super().__init__(
            params=params,
            lr=args.lr,
            betas=adam_betas,
            weight_decay=args.weight_decay,
            amsgrad=args.use_amsgrad,
        )

    @staticmethod
    def add_args(parser) -> None:
        parser.add_argument(
            "--use_amsgrad",
            action="store_true",
            default=False,
            help="whether to use the AMSGrad variant of Adam from the paper On the Convergence of Adam and Beyond",
        )
        parser.add_argument(
            "--adam_betas",
            type=float,
            nargs=2,
            default=[0.9, 0.999],
            help="momentume values for Adam optimizer",
        )


@register_object("adamw", "optimizer")
class AdamW(optim.AdamW, Nox):
    """
    https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW
    """

    def __init__(self, params, args):
        adam_betas = tuple(args.adam_betas)
        super().__init__(
            params=params,
            lr=args.lr,
            betas=adam_betas,
            weight_decay=args.weight_decay,
            amsgrad=args.use_amsgrad,
        )

    @staticmethod
    def add_args(parser) -> None:
        parser.add_argument(
            "--use_amsgrad",
            action="store_true",
            default=False,
            help="whether to use the AMSGrad variant of Adam from the paper On the Convergence of Adam and Beyond",
        )
        parser.add_argument(
            "--adam_betas",
            type=float,
            nargs=2,
            default=[0.9, 0.999],
            help="momentume values for Adam optimizer",
        )
