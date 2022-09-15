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


@register_object("graph_base", "lightning")
class GraphBase(Base):
    """
    Args:
        args: argparser Namespace
    """

    def __init__(self, args):
        super(GraphBase, self).__init__()

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        output = super(GraphBase, self).training_step(batch, batch_idx, optimizer_idx)
        batch["gsm"] = self.trainer.train_dataloader.dataset.datasets.split_graph.to(
            self.device
        )
        return output

    def validation_step(self, batch, batch_idx, optimizer_idx=None):
        output = super(GraphBase, self).validation_step(batch, batch_idx, optimizer_idx)

        batch["gsm"] = self.trainer.val_dataloaders[0].dataset.split_graph.to(
            self.device
        )
        return output

    def test_step(self, batch, batch_idx):
        output = super(GraphBase, self).test_step(batch, batch_idx)

        batch["gsm"] = self.trainer.test_dataloaders[0].dataset.split_graph.to(
            self.device
        )
        return output
