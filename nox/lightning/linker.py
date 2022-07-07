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
from nox.utils.nbf import nbf_utils


@register_object("linker", "lightning")
class Linker(Base):
    """
    PyTorch Lightning module used as base for running training and test loops

    Args:
        args: argparser Namespace
    """

    def __init__(self, args):
        super(Linker, self).__init__(args)
        self.num_negative = args.num_negative
        self.strict_negative = args.strict_negative

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        """
        Single training step
        """
        self.phase = "train"
        batch["graph"] = self.trainer.train_dataloader.dataset.datasets.split_graph.to(self.device)

        batch["triplet"] = nbf_utils.negative_sampling(
            batch["graph"],
            batch["triplet"],
            self.num_negative,
            strict=self.strict_negative,
        )

        output = self.step(batch, batch_idx, optimizer_idx)
        return output

    def validation_step(self, batch, batch_idx, optimizer_idx=None):
        """
        Single validation step
        """
        self.phase = "val"
        output = dict()

        batch["graph"] = self.trainer.val_dataloaders[0].dataset.split_graph.to(self.device)

        t_triplet, h_triplet = nbf_utils.all_negative(batch["graph"], batch["triplet"])

        t_batch = {"triplet": t_triplet, "graph": batch["graph"]}
        h_batch = {"triplet": h_triplet, "graph": batch["graph"]}

        t_output = self.step(t_batch, batch_idx, optimizer_idx)
        h_output = self.step(h_batch, batch_idx, optimizer_idx)

        filtered_data = getattr(
            self.trainer.val_dataloaders[0], "filtered_data", batch["graph"]
        )

        t_mask, h_mask = nbf_utils.strict_negative_mask(filtered_data, batch["triplet"])

        pos_h_index, pos_t_index, _ = batch["triplet"].t()

        t_ranking = nbf_utils.compute_ranking(
            t_output["model_output"]["logit"], pos_t_index, t_mask
        )
        h_ranking = nbf_utils.compute_ranking(
            h_output["model_output"]["logit"], pos_h_index, h_mask
        )

        num_t_negative = t_mask.sum(dim=-1)
        num_h_negative = h_mask.sum(dim=-1)

        output["preds_dict"] = {
            "rankings": [t_ranking, h_ranking],
            "num_negatives": [num_t_negative, num_h_negative],
        }
        output["loss"] = (t_output["loss"] + h_output["loss"]) / 2

        return output

    def test_step(self, batch, batch_idx):
        """
        Single testing step

        * save_predictions will save the dictionary output['preds_dict'], which typically includes sample_ids, probs, predictions, etc.
        * save_hiddens: will save the value of output['preds_dict']['hidden']
        """
        self.phase = "test"

        output = dict()

        batch["graph"] = self.trainer.test_dataloaders[0].dataset.split_graph.to(self.device)

        t_triplet, h_triplet = nbf_utils.all_negative(batch["graph"], batch["triplet"])

        t_batch = {"triplet": t_triplet, "graph": batch["graph"]}
        h_batch = {"triplet": h_triplet, "graph": batch["graph"]}

        t_output = self.forward(t_batch, batch_idx)
        h_output = self.forward(h_batch, batch_idx)

        filtered_data = getattr(
            self.trainer.test_dataloaders[0], "filtered_data", batch["graph"]
        )

        t_mask, h_mask = nbf_utils.strict_negative_mask(filtered_data, batch["triplet"])

        pos_h_index, pos_t_index, _ = batch["triplet"].t()

        t_ranking = nbf_utils.compute_ranking(
            t_output["preds_dict"]["logit"], pos_t_index, t_mask
        )
        h_ranking = nbf_utils.compute_ranking(
            h_output["preds_dict"]["logit"], pos_h_index, h_mask
        )

        num_t_negative = t_mask.sum(dim=-1)
        num_h_negative = h_mask.sum(dim=-1)

        output["preds_dict"] = {
            "rankings": [t_ranking, h_ranking],
            "num_negatives": [num_t_negative, num_h_negative],
            "t_logit": t_output["preds_dict"]["logit"],
            "h_logit": h_output["preds_dict"]["logit"],
        }
        output["loss"] = (t_output["loss"] + h_output["loss"]) / 2

        if self.args.save_predictions:
            self.save_predictions(output["preds_dict"])
        elif self.args.save_hiddens:
            self.save_hiddens(output["preds_dict"])
        output = {k: v for k, v in output.items() if k not in self.UNLOG_KEYS}
        output["preds_dict"] = {
            k: v for k, v in output["preds_dict"].items() if k not in self.UNLOG_KEYS
        }
        return output

