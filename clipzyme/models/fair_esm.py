import torch
import torch.nn as nn
import copy
from typing import List
from clipzyme.models.abstract import AbstractModel
from clipzyme.utils.classes import set_nox_type
from clipzyme.utils.registry import register_object, get_object
from clipzyme.utils.amino_acids import AA_TO_SMILES
from torch_geometric.data import Data, HeteroData, Batch
from clipzyme.utils.pyg import from_smiles
from clipzyme.utils.smiles import get_rdkit_feature


@register_object("fair_esm", "model")
class FairEsm(AbstractModel):
    """
    Refer to https://github.com/facebookresearch/esm#available-models
    """

    def __init__(self, args):
        super(FairEsm, self).__init__()
        self.args = args
        torch.hub.set_dir(args.pretrained_hub_dir)
        self.model, self.alphabet = torch.hub.load(
            "facebookresearch/esm:v2.0.0", args.esm_name
        )
        self.batch_converter = self.alphabet.get_batch_converter()
        self.register_buffer("devicevar", torch.zeros(1, dtype=torch.int8))
        if args.freeze_esm:
            self.model.eval()

        self.repr_layer = args.esm_hidden_layer
        self.use_cls_token = args.use_esm_cls_token

    def forward(self, x):
        """
        x: list of str (protein sequences)
        """
        output = {}
        if isinstance(x, list):
            pass
        elif isinstance(x, dict):
            try:
                x = x["sequence"]
            except:
                raise ValueError(
                    "FairEsm forward received dict without 'sequence' key "
                )

        fair_x = self.truncate_protein(x)
        batch_labels, batch_strs, batch_tokens = self.batch_converter(fair_x)
        batch_tokens = batch_tokens.to(self.devicevar.device)

        if self.args.freeze_esm:
            self.model.requires_grad_(False)
            with torch.no_grad():
                result = self.model(
                    batch_tokens,
                    repr_layers=[self.repr_layer],
                    return_contacts=self.args.esm_return_contacts,
                )
        else:
            result = self.model(
                batch_tokens,
                repr_layers=[self.repr_layer],
                return_contacts=self.args.esm_return_contacts,
            )

        # Generate per-sequence representations via averaging
        if self.use_cls_token:
            output["hidden"] = result["representations"][self.repr_layer][0]
        else:
            # remove cls, eos, and padding embeddings
            sequence_mask = torch.ne(batch_tokens, self.alphabet.cls_idx).long()
            sequence_mask *= torch.ne(batch_tokens, self.alphabet.eos_idx).long()
            sequence_mask *= torch.ne(batch_tokens, self.alphabet.padding_idx).long()
            sequence_mask = sequence_mask.unsqueeze(-1)
            # remove cls and eos tokens
            output["hidden"] = (
                result["representations"][self.repr_layer] * sequence_mask
            ).sum(1) / sequence_mask.sum(1)
            output["mask_hiddens"] = sequence_mask

        output["tokens"] = batch_tokens
        output["token_hiddens"] = result["representations"][self.repr_layer]

        return output

    def truncate_protein(self, x, max_length=1024):
        # max length allowed is 1024
        return [
            (i, s[: 1024 - 2]) if not isinstance(x[0], list) else (i, s[0][: 1024 - 2])
            for i, s in enumerate(x)
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
            default="/home/snapshots/metabolomics",
            help="directory to torch hub where pretrained models are saved",
        )
        parser.add_argument(
            "--esm_name",
            type=str,
            default="esm2_t12_35M_UR50D",
            help="directory to torch hub where pretrained models are saved",
        )
        parser.add_argument(
            "--freeze_esm",
            action="store_true",
            default=False,
            help="do not update encoder weights",
        )
        parser.add_argument(
            "--esm_hidden_layer",
            type=int,
            default=12,
            help="do not update encoder weights",
        )
        parser.add_argument(
            "--use_esm_cls_token",
            action="store_true",
            default=False,
            help="use cls token as representation",
        )
        parser.add_argument(
            "--esm_return_contacts",
            action="store_true",
            default=False,
            help="return contacts",
        )


@register_object("fair_esm2", "model")
class FairEsm2(FairEsm):
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
        cargs.mlp_input_dim = args.protein_hidden_dim
        args.freeze_esm = args.freeze_encoder
        self.mlp = get_object(args.protein_classifer, "model")(cargs)
        if self.args.freeze_encoder:
            self.encoder.eval()

    def forward(self, batch):
        output = {}
        if self.args.freeze_encoder:
            self.encoder.requires_grad_(False)
            with torch.no_grad():
                output_esm = self.encoder(batch["x"])
        else:
            output_esm = self.encoder(batch["x"])
        # output["protein_hidden"] = output_esm["hidden"]
        output.update(self.mlp({"x": output_esm["hidden"]}))
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
            default=False,
            help="do not update encoder weights",
        )
        parser.add_argument(
            "--protein_hidden_dim",
            type=int,
            default=480,
            help="hidden dimension of the protein",
        )
        parser.add_argument(
            "--protein_classifer",
            type=str,
            default="mlp_classifier",
            help="name of classifier",
            action=set_nox_type("model"),
        )
