import torch
import torch.nn as nn
import copy
from nox.models.abstract import AbstractModel
from nox.utils.classes import set_nox_type
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
        if not self.args.train_encoder:
            self.model.eval()

    def forward(self, x, batch=None):
        output = {}
        fair_x = self.truncate_protein(x)
        batch_labels, batch_strs, batch_tokens = self.batch_converter(fair_x)
        batch_tokens = batch_tokens.to(self.devicevar.device)
        repr_layer = 33  # TODO: add arg?

        if not self.args.train_encoder:
            with torch.no_grad():
                result = self.model(
                    batch_tokens, repr_layers=[repr_layer], return_contacts=False
                )
        else:
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

        output["hidden"] = torch.stack(hiddens)

        return output
    
    def truncate_protein(self, x, max_length=1024):
        # max length allowed is 1024
        return [
            (i, s[: 1024 - 2]) if not isinstance(x[0], list) else (i, s[0][: 1024 - 2]) for i, s in enumerate(x) 
        ]  

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--pretrained_hub_dir",
            type=str,
            default="/Mounts/rbg-storage1/snapshots/metabolomics",
            help="directory to torch hub where pretrained models are saved",
        )
        parser.add_argument(
            "--train_encoder",
            action="store_true",
            default=False,
            help="do not update encoder weights",
        )

@register_object("fair_esm2", "model")
class FairEsm2(FairEsm):
    def __init__(self, args):
        super(FairEsm2, self).__init__(args)
        self.args = args
        torch.hub.set_dir(args.pretrained_hub_dir)
        self.model, self.alphabet = torch.hub.load(
            "facebookresearch/esm:main", "esm2_t33_650M_UR50D"
        )
        self.batch_converter = self.alphabet.get_batch_converter()
        self.register_buffer("devicevar", torch.zeros(1, dtype=torch.int8))
        if not self.args.train_encoder:
            self.model.eval()

    def truncate_protein(self, x, max_length=torch.inf):
        return [
            (i, s) if not isinstance(x[0], list) else (i, s[0]) for i, s in enumerate(x) 
        ]  

@register_object("protein_encoder", "model")
class ProteinEncoder(AbstractModel):
    def __init__(self, args):
        super(ProteinEncoder, self).__init__()
        self.args = args
        self.encoder = get_object(args.protein_encoder_type, "model")(args)
        cargs = copy.deepcopy(args)
        cargs.mlp_input_dim = 1280  # TODO: add arg?
        self.mlp = get_object("mlp_classifier", "model")(cargs)

    def forward(self, x, batch=None):
        output = {}
        if self.args.freeze_encoder:
            with torch.no_grad():
                output_esm = self.encoder(x, batch)
        else:
            output_esm = self.encoder(x, batch)
        output["protein_hidden"] = output_esm["hidden"]
        output["hidden"] = self.mlp(output_esm["hidden"])["hidden"]
        return output

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--protein_encoder_type",
            type=str,
            default="fair_esm",
            help="name of the protein encoder",
            action=set_nox_type("model"),
        )
        parser.add_argument(
            "--freeze_encoder",
            action="store_true",
            default=True,
            help="do not update encoder weights",
        )
