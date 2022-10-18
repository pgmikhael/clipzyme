import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import pickle
import os
from nox.utils.registry import get_object, register_object
from nox.utils.classes import Nox, set_nox_type
from nox.lightning.base import Base

@register_object("step_base", "lightning")
class StepBase(Base):
    """
    PyTorch Lightning module used as base for running training and test loops

    Compute metrics every step

    Args:
        args: argparser Namespace
    """

    def step(self, batch, batch_idx, optimizer_idx):
        """
        Defines a single training or validation step:
            Computes losses given batch and model outputs

        Returns:
            logged_output: dict with losses and predictions

        Args:
            batch: dict obtained from DataLoader. batch must contain they keys ['x', 'sample_id']
        """
        logged_output = OrderedDict()
        model_output = self.model(batch)
        loss, logging_dict, predictions_dict = self.compute_loss(model_output, batch)
        predictions_dict = self.store_in_predictions(predictions_dict, batch)
        predictions_dict = self.store_in_predictions(predictions_dict, model_output)

        logged_output["loss"] = loss
        logged_output.update(logging_dict)
        logged_output["preds_dict"] = predictions_dict

        if (
            (self.args.log_gen_image)
            and (self.trainer.is_global_zero)
            and (batch_idx == 0)
            and (self.current_epoch % 100 == 0)
        ):
            self.log_image(model_output, batch)

        return logged_output

    def forward(self, batch, batch_idx=0):
        """
        Forward defines the prediction/inference actions
            Similar to self.step() but also allows for saving predictions and hiddens
            Computes losses given batch and model outputs

        Returns:
            logged_output: dict with losses and predictions

        Args:
            batch: dict obtained from DataLoader. batch must contain they keys ['x', 'sample_id']
        """
        logged_output = OrderedDict()
        model_output = self.model(batch)
        if not self.args.predict:
            loss, logging_dict, predictions_dict = self.compute_loss(
                model_output, batch
            )
            predictions_dict = self.store_in_predictions(predictions_dict, batch)
        predictions_dict = self.store_in_predictions(predictions_dict, model_output)
        logged_output["loss"] = loss
        logged_output.update(logging_dict)
        logged_output["preds_dict"] = predictions_dict
        if self.args.save_hiddens:
            logged_output["preds_dict"].update(model_output)

        if (self.args.log_gen_image) and (batch_idx == 0):
            self.log_image(model_output, batch)
    
        return logged_output

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        """
        Single training step
        """
        self.phase = "train"
        output = self.step(batch, batch_idx, optimizer_idx)

        metrics = self.compute_metric(output["preds_dict"])
        output.update(metrics)
        self.log_outputs(output, "train")

        return {"loss": output["loss"]}

    def validation_step(self, batch, batch_idx, optimizer_idx=None):
        """
        Single validation step
        """
        self.phase = "val"
        output = self.step(batch, batch_idx, optimizer_idx)

        metrics = self.compute_metric(output["preds_dict"])
        output.update(metrics)
        self.log_outputs(output, "val")
        return {"loss": output["loss"]}

    def test_step(self, batch, batch_idx):
        """
        Single testing step

        * save_predictions will save the dictionary output['preds_dict'], which typically includes sample_ids, probs, predictions, etc.
        * save_hiddens: will save the value of output['preds_dict']['hidden']
        """
        self.phase = "test"
        output = self.forward(batch, batch_idx)
        if self.args.save_predictions:
            self.save_predictions(output["preds_dict"])
        elif self.args.save_hiddens:
            self.save_hiddens(output["preds_dict"])
        output = {k: v for k, v in output.items() if k not in self.UNLOG_KEYS}
        output["preds_dict"] = {
            k: v for k, v in output["preds_dict"].items() if k not in self.UNLOG_KEYS
        }

        metrics = self.compute_metric(output["preds_dict"])
        output.update(metrics)
        self.log_outputs(output, "test")

        return {"loss": output["loss"]}
    