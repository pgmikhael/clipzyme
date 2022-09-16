import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import pickle
import os
from nox.lightning.base import Base
from nox.utils.registry import get_object, register_object
from nox.utils.classes import Nox, set_nox_type


@register_object("gsm_base", "lightning")
class GSMBase(Base):
    """
    Args:
        args: argparser Namespace
    """

    def __init__(self, args):
        super(GSMBase, self).__init__(args)

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        batch["gsm"] = self.trainer.train_dataloader.dataset.datasets.split_graph.to(
            self.device
        )
        batch["y"] = batch["mol"].y
        output = super(GSMBase, self).training_step(batch, batch_idx, optimizer_idx)
        return output

    def validation_step(self, batch, batch_idx, optimizer_idx=None):
        batch["gsm"] = self.trainer.val_dataloaders[0].dataset.split_graph.to(
            self.device
        )
        batch["y"] = batch["mol"].y
        output = super(GSMBase, self).validation_step(batch, batch_idx, optimizer_idx)
        return output

    def test_step(self, batch, batch_idx):
        batch["gsm"] = self.trainer.test_dataloaders[0].dataset.split_graph.to(
            self.device
        )
        batch["y"] = batch["mol"].y
        output = super(GSMBase, self).test_step(batch, batch_idx)
        return output
