import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import pickle
import os
from nox.utils.registry import get_object, register_object
from nox.utils.classes import Nox, set_nox_type


@register_object("base", "lightning")
class Base(pl.LightningModule, Nox):
    """
    PyTorch Lightning module used as base for running training and test loops

    Args:
        args: argparser Namespace
    """

    def __init__(self, args):
        super(Base, self).__init__()
        self.save_hyperparameters()
        self.args = args
        self.model = get_object(args.model_name, "model")(args)

    def setup(self, stage):
        self.loss_fns = {
            "train": [get_object(l, "loss")() for l in self.args.loss_names]
        }
        self.loss_fns["val"] = self.loss_fns["train"]
        self.loss_fns["test"] = (
            self.loss_fns["train"]
            if self.args.loss_names_for_eval is None
            else [get_object(l, "loss")() for l in self.args.loss_names_for_eval]
        )
        self.metrics = [
            get_object(m, "metric")(self.args) for m in self.args.metric_names
        ]
        self.metric_keys = list(
            set([key for metric in self.metrics for key in metric.metric_keys])
        )

    @property
    def LOG_KEYS(self):
        return [
            "loss",
            "accuracy",
            "mean",
            "std",
            "precision",
            "recall",
            "f1",
            "auc",
            "similarity",
            "tau",
            "mse",
            "mae",
            "r2",
            "c_index",
            "hit",
        ]

    @property
    def UNLOG_KEYS(self):
        default = ["activ", "hidden"]
        keys_to_unlog = []
        for k in default:
            if k not in self.metric_keys:
                keys_to_unlog.append(k)
        return keys_to_unlog

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
        return output

    def validation_step(self, batch, batch_idx, optimizer_idx=None):
        """
        Single validation step
        """
        self.phase = "val"
        output = self.step(batch, batch_idx, optimizer_idx)
        return output

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
        return output

    def training_epoch_end(self, outputs):
        """
        End of single training epoch
            - Aggregates predictions and losses from all steps
            - Computes the metric (auc, accuracy, etc.)
        """
        if len(outputs) == 0:
            return
        outputs = gather_step_outputs(outputs)
        outputs["loss"] = outputs["loss"].mean()
        if "preds_dict" in outputs:
            outputs.update(self.compute_metric(outputs["preds_dict"]))
            self.log_outputs(outputs, "train")
        return

    def validation_epoch_end(self, outputs):
        """
        End of single validation epoch
            - Aggregates predictions and losses from all steps
            - Computes the metric (auc, accuracy, etc.)
        """
        if len(outputs) == 0:
            return
        outputs = gather_step_outputs(outputs)
        outputs["loss"] = outputs["loss"].mean()
        if "preds_dict" in outputs:
            outputs.update(self.compute_metric(outputs["preds_dict"]))
            self.log_outputs(outputs, "val")
        return

    def test_epoch_end(self, outputs):
        """
        End of testing
            - Aggregates predictions and losses from all batches
            - Computes the metric if defined in args
        """
        if len(outputs) == 0:
            return
        outputs = gather_step_outputs(outputs)
        if isinstance(outputs.get("loss", 0), torch.Tensor):
            outputs["loss"] = outputs["loss"].mean()
        if "preds_dict" in outputs:
            if not self.args.predict:
                outputs.update(self.compute_metric(outputs["preds_dict"]))
            self.log_outputs(outputs, "test")
        return

    def configure_optimizers(self):
        """
        Obtain optimizers and hyperparameter schedulers for model

        """
        optimizer = get_object(self.args.optimizer_name, "optimizer")(
            self.parameters(), self.args
        )
        schedule = get_object(self.args.scheduler_name, "scheduler")(
            optimizer, self.args
        )

        scheduler = {
            "scheduler": schedule,
            "monitor": self.args.monitor,
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    def compute_loss(self, model_output, batch):
        """
        Compute model loss:
            Iterates through loss functions defined in args and computes losses and predictions
            Adds losses and stores predictions for batch in dictionary

        Returns:
            total_loss (torch.Tensor): aggregate loss value that is propagated backwards for gradient computation
            logging_dict: dict of losses (and other metrics)
            predictions: dict of predictions (preds, probs, etc.)
        """
        total_loss = 0
        logging_dict, predictions = OrderedDict(), OrderedDict()
        for loss_fn in self.loss_fns[self.phase]:
            loss, l_dict, p_dict = loss_fn(model_output, batch, self, self.args)
            total_loss += loss
            logging_dict.update(l_dict)
            predictions.update(p_dict)
        return total_loss, logging_dict, predictions

    def compute_metric(self, predictions):
        logging_dict = OrderedDict()
        for metric_fn in self.metrics:
            l_dict = metric_fn(predictions, self.args)
            logging_dict.update(l_dict)
        return logging_dict

    def store_in_predictions(self, preds, storage_dict):
        for m in get_object(self.args.dataset_name, "dataset").DATASET_ITEM_KEYS:
            if m in storage_dict:
                preds[m] = storage_dict[m]

        for m in self.metric_keys:
            if m in storage_dict:
                if torch.is_tensor(storage_dict[m]) and storage_dict[m].requires_grad:
                    preds[m] = storage_dict[m].detach()
                else:
                    preds[m] = storage_dict[m]
        return preds

    def log_outputs(self, outputs, key):
        """
        Compute performance metrics after epoch ends:
            Iterates through metric functions defined in args and computes metrics
            Logs the metric values into logger (Comet, Tensorboard, etc.)
        """
        logging_dict = {}
        for k, v in outputs.items():
            if isinstance(v, torch.Tensor) and any([i in k for i in self.LOG_KEYS]):
                logging_dict["{}_{}".format(key, k)] = v.mean()
        # log clocktime of methods for epoch
        if (self.args.profiler is not None) and (self.args.log_profiler):
            logging_dict.update(self.get_time_profile(key))
        self.log_dict(logging_dict, batch_size = 1, prog_bar=True, logger=True)

    def get_time_profile(self, key):
        """Obtain trainer method times

        Args:
            key (str): one of ['train', 'val', 'test]

        Returns:
            dict: mean of clocktime of each method for past epoch
        """
        if key == "train":
            num_steps = self.trainer.num_training_batches
        if key == "val":
            num_steps = self.trainer.num_val_batches[0]
        if key == "test":
            num_steps = self.trainer.num_test_batches[0]

        time_profile = {}
        for k, v in self.trainer.profiler.recorded_durations.items():
            time_profile[k] = np.mean(v[-num_steps:])
        return time_profile

    def save_predictions(self, outputs):
        """
        Saves model predictions as pickle files
            Makes a directory under /inference_dir/experiment_name/
            Stores predictions for each sample individually under /inference_dir/experiment_name/sample_[sample_id].predictions

        * Requires outputs to contain the keys ['sample_id']
        """
        experiment_name = (
            os.path.splitext(os.path.basename(self.args.checkpoint_path))[0]
            if (self.args.from_checkpoint and not self.args.train)
            else self.args.experiment_name
        )
        for idx, sampleid in enumerate(outputs["exam"]):
            sampledict = {
                k: v[idx]
                for k, v in outputs.items()
                if (len(v) == len(outputs["exam"]))
            }
            for k, v in sampledict.items():
                if isinstance(v, torch.Tensor) and v.is_cuda:
                    sampledict[k] = v.cpu()
            predictions_filename = os.path.join(
                self.args.inference_dir,
                experiment_name,
                "sample_{}.predictions".format(sampleid),
            )
            dump_pickle(sampledict, predictions_filename)

    def save_hiddens(self, outputs):
        """
        Saves the model's hidden layer outputs as pickle files
            Makes a directory under /inference_dir/experiment_name/
            Stores predictions for each sample individually under /inference_dir/experiment_name/sample_[sample_id].hiddens

        * Requires outputs to contain the keys ['sample_id', 'hidden]
        """
        experiment_name = (
            os.path.splitext(os.path.basename(self.args.checkpoint_path))[0]
            if (self.args.from_checkpoint and not self.args.train)
            else self.args.experiment_name
        )
        idx = outputs["exam"]
        # hiddens = nn.functional.normalize(outputs['hidden'], dim = 1)
        hiddens = [
            {
                k: v[i].cpu() if v.is_cuda else v[i]
                for k, v in outputs.items()
                if ("hidden" in k) and (len(v) == len(idx))
            }
            for i in range(len(idx))
        ]
        for i, h in zip(idx, hiddens):
            predictions_filename = os.path.join(
                self.args.inference_dir, experiment_name, "sample_{}.hiddens".format(i)
            )
            dump_pickle(h, predictions_filename)

    def log_image(self, model_output, batch):
        # log one sample from each epoch
        sid = batch["exam"][0]
        for k, v in model_output.items():
            if "reconstruction" in k:
                img = model_output[k][0].detach().cpu()
                if img.shape[0] != 3:
                    img = img.numpy()
                    for cid, chan in enumerate(img):
                        self.logger.log_image(
                            chan,
                            "Sample{}_{}_Chan{}_Epoch{}_Step{}".format(
                                sid, k, cid, self.current_epoch, self.global_step
                            ),
                        )
                else:
                    img = img.permute(1, 2, 0).numpy()
                    self.logger.log_image(
                        img,
                        "Sample{}_{}_Epoch{}_Step{}".format(
                            sid, k, self.current_epoch, self.global_step
                        ),
                    )

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--model_name",
            type=str,
            action=set_nox_type("model"),
            default="classifier",
            help="Name of parent model",
        )


def gather_step_outputs(outputs):
    """
    Collates the dictionary outputs from each step into a single dictionary

    Returns:
        output_dict (dict): dictionary mapping step output keys to lists or tensors
    """

    output_dict = OrderedDict()
    if isinstance(outputs[-1], list):  # adversarial setting with two optimizers
        outputs = outputs[0]

    for k in outputs[-1].keys():
        if k == "preds_dict":
            output_dict[k] = gather_step_outputs(
                [output["preds_dict"] for output in outputs]
            )
        elif (
            isinstance(outputs[-1][k], torch.Tensor) and len(outputs[-1][k].shape) == 0
        ):
            output_dict[k] = torch.stack([output[k] for output in outputs])
        elif isinstance(outputs[-1][k], torch.Tensor):
            output_dict[k] = torch.cat([output[k] for output in outputs], dim=0)
        else:
            output_dict[k] = [output[k] for output in outputs]
    return output_dict


def dump_pickle(file_obj, file_name):
    """
    Saves object as a binary pickle file
        Creates directory of file
        Saves file

    Args:
        file_obj: object
        file_name: path to file
    """
    if not os.path.exists(os.path.dirname(file_name)):
        try:
            os.makedirs(os.path.dirname(file_name))
        except OSError as exc:  # Guard against race condition
            if exc.errno != exc.errno.EEXIST:
                raise
    pickle.dump(file_obj, open(file_name, "wb"))
