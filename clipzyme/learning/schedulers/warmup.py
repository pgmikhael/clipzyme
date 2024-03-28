"""
Copyright (c) 2022, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import math
from clipzyme.utils.registry import register_object
from clipzyme.utils.classes import Nox
from torch.optim.lr_scheduler import _LRScheduler


@register_object("linear_warmup_step_lr", "scheduler")
class LinearWarmupStepLRScheduler(_LRScheduler, Nox):
    def __init__(self, optimizer, args):
        max_epoch = args.max_epochs
        min_lr = args.warmup_min_lr
        init_lr = args.warmup_init_lr
        decay_rate = args.warmup_decay_rate
        warmup_start_lr = args.warmup_start_lr
        warmup_steps = args.warmup_steps

        self.optimizer = optimizer

        self.max_epoch = max_epoch
        self.min_lr = min_lr

        self.decay_rate = decay_rate

        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr
        super().__init__(optimizer)

    def step(self, cur_epoch=0):
        self._step_count += 1
        cur_step = self._step_count
        if cur_step < self.warmup_steps:
            warmup_lr_schedule(
                step=cur_step,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                init_lr=self.warmup_start_lr,
                max_lr=self.init_lr,
            )
        else:
            step_lr_schedule(
                epoch=cur_epoch,
                optimizer=self.optimizer,
                init_lr=self.init_lr,
                min_lr=self.min_lr,
                decay_rate=self.decay_rate,
            )

    @staticmethod
    def add_args(parser) -> None:
        parser.add_argument(
            "--warmup_min_lr",
            type=float,
            default=None,
            help="minimum learning rate",
        )
        parser.add_argument(
            "--warmup_init_lr",
            type=float,
            default=None,
            help="base learning rate",
        )
        parser.add_argument(
            "--warmup_decay_rate",
            type=float,
            default=1,
            help="decay rate",
        )
        parser.add_argument(
            "--warmup_start_lr",
            type=float,
            default=-1,
            help="lr of warmup",
        )
        parser.add_argument(
            "--warmup_steps",
            type=float,
            default=0,
            help="step of warmpu",
        )


@register_object("linear_warmup_cosine_lr", "scheduler")
class LinearWarmupCosineLRScheduler(_LRScheduler, Nox):
    def __init__(self, optimizer, args):
        max_epoch = args.max_epochs
        min_lr = args.warmup_min_lr
        init_lr = args.warmup_init_lr
        warmup_start_lr = args.warmup_start_lr
        warmup_steps = args.warmup_steps

        self.optimizer = optimizer

        self.max_epoch = max_epoch
        self.min_lr = min_lr

        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr
        super().__init__(optimizer)

    def step(self, cur_epoch=0):
        # assuming the warmup iters less than one epoch
        self._step_count += 1
        cur_step = self._step_count
        if cur_step < self.warmup_steps:
            warmup_lr_schedule(
                step=cur_step,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                init_lr=self.warmup_start_lr,
                max_lr=self.init_lr,
            )
        else:
            cosine_lr_schedule(
                epoch=cur_epoch,
                optimizer=self.optimizer,
                max_epoch=self.max_epoch,
                init_lr=self.init_lr,
                min_lr=self.min_lr,
            )

    @staticmethod
    def add_args(parser) -> None:
        parser.add_argument(
            "--warmup_min_lr",
            type=float,
            default=None,
            help="minimum learning rate",
        )
        parser.add_argument(
            "--warmup_init_lr",
            type=float,
            default=None,
            help="base learning rate",
        )
        parser.add_argument(
            "--warmup_start_lr",
            type=float,
            default=-1,
            help="lr of warmup",
        )
        parser.add_argument(
            "--warmup_steps",
            type=float,
            default=0,
            help="step of warmpu",
        )


@register_object("constant_lr", "scheduler")
class ConstantLRScheduler(_LRScheduler, Nox):
    def __init__(self, optimizer, args):
        init_lr = args.init_lr
        warmup_start_lr = args.warmup_start_lr
        warmup_steps = args.warmup_steps

        self.optimizer = optimizer
        self.lr = init_lr
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def step(self, cur_epoch=0):
        cur_step = self._step_count

        if self.warmup_steps < cur_step:
            warmup_lr_schedule(
                step=cur_step,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                init_lr=self.warmup_start_lr,
                max_lr=self.lr,
            )
        else:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr

    @staticmethod
    def add_args(parser) -> None:
        parser.add_argument(
            "--warmup_init_lr",
            type=float,
            default=None,
            help="base learning rate",
        )
        parser.add_argument(
            "--warmup_start_lr",
            type=float,
            default=-1,
            help="lr of warmup",
        )
        parser.add_argument(
            "--warmup_steps",
            type=float,
            default=0,
            help="step of warmpu",
        )


def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (
        1.0 + math.cos(math.pi * epoch / max_epoch)
    ) + min_lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
    """Warmup the learning rate"""
    lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max(max_step, 1))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def step_lr_schedule(optimizer, epoch, init_lr, min_lr, decay_rate):
    """Decay the learning rate"""
    lr = max(min_lr, init_lr * (decay_rate**epoch))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
