import torch
from torch import optim
from nox.utils.registry import register_object
from nox.utils.classes import Nox
from torch.optim.lr_scheduler import _LRScheduler


@register_object("noam", "scheduler")
class NoamLR(_LRScheduler, Nox):
    """
    Learning rate scheduler as in Transformer paper (AIAYN)

    Adapted / corrected from https://github.com/tugstugi/pytorch-saltnet/blob/master/utils/lr_scheduler.py
    """

    def __init__(self, optimizer, args):
        self.warmup_steps = args.noam_warmup_steps
        self.hidden_dim = args.noam_model_hidden_dim
        self.base_lr = args.noam_base_lr
        self._step = 0
        super().__init__(optimizer)

    def get_lr(self):
        self._step += 1
        scale = self.hidden_dim ** (-0.5) * min(
            self._step ** (-0.5), self._step * self.warmup_steps ** (-1.5)
        )
        return [base_lr * scale for base_lr in [self.base_lr] * len(self.base_lrs)]

    @staticmethod
    def add_args(parser) -> None:
        parser.add_argument(
            "--noam_base_lr",
            type=float,
            default=2,
            help="base learning rate to use after first step",
        )
        parser.add_argument(
            "--noam_warmup_steps",
            type=int,
            default=8000,
            help="number of steps to use for warmup",
        )
        parser.add_argument(
            "--noam_model_hidden_dim",
            type=int,
            default=None,
            required=True,
            help="model dimensions",
        )
