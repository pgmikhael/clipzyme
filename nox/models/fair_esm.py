import torch
import torch.nn as nn
import copy
from nox.models.abstract import AbstractModel
from nox.utils.registry import register_object, get_object


@register_object("fair_esm", "model")
class FairEsm(AbstractModel):
    def __init__(self, args):
        super(FairEsm, self).__init__()
        self.args = args
        torch.hub.set_dir(args.pretrained_hub_dir)
        self.model, self.alphabet = torch.hub.load(
            "facebookresearch/esm:main", "esm1b_t33_650M_UR50S"
        )
        self.batch_converter = self.alphabet.get_batch_converter()
        self.register_buffer("devicevar", torch.zeros(1, dtype=torch.int8))

    def forward(self, x, batch=None):
        output = {}
        fair_x = [
            (i, s[: 1024 - 2]) for i, s in enumerate(x)
        ]  # max length allowed is 1024
        batch_labels, batch_strs, batch_tokens = self.batch_converter(fair_x)
        batch_tokens = batch_tokens.to(self.devicevar.device)
        repr_layer = 33  # TODO: add arg?
        result = self.model(
            batch_tokens, repr_layers=[repr_layer], return_contacts=False
        )

        # Generate per-sequence representations via averaging
        hiddens = []
        for sample_num, sample in enumerate(x):
            hiddens.append(
                result["representations"][repr_layer][
                    sample_num, 1 : len(sample) + 1
                ].mean(0)
            )

        output["protein_hidden"] = torch.stack(hiddens)

        return output

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--pretrained_hub_dir",
            type=str,
            default="/Mounts/rbg-storage1/users/pgmikhael/torchhub",
            help="directory to torch hub where pretrained models are saved",
        )


@register_object("protein_encoder", "model")
class ProteinEncoder(AbstractModel):
    def __init__(self, args):
        super(ProteinEncoder, self).__init__()
        self.args = args
        self.encoder = get_object("fair_esm", "model")(args)
        args.mlp_input_dim = 1280  # TODO: add arg?
        self.mlp = get_object("mlp_classifier", "model")(args)

    def forward(self, x, batch=None):
        output = {}
        if self.args.freeze_encoder:
            with torch.no_grad():
                output_esm = self.encoder(x, batch)
        else:
            output_esm = self.encoder(x, batch)
        output["protein_hidden"] = output_esm["protein_hidden"]
        output["hidden"] = self.mlp(output_esm["protein_hidden"])["hidden"]
        return output

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--freeze_encoder",
            action="store_true",
            default=True,
            help="do not update encoder weights",
        )
