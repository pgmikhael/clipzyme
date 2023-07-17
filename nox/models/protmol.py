import torch
import torch.nn as nn
import copy
import inspect
import torch.nn.functional as F
from typing import Union, Tuple, Any, List, Dict, Optional
from nox.utils.classes import set_nox_type
from nox.models.abstract import AbstractModel
from nox.utils.registry import register_object, get_object
from nox.utils.smiles import standardize_reaction, tokenize_smiles
from transformers import (
    EncoderDecoderModel,
    EncoderDecoderConfig,
    BertConfig,
    BertTokenizer,
    AutoTokenizer,
    AutoModel,
    EsmModel,
    AutoModelForCausalLM,
)
from transformers.modeling_outputs import BaseModelOutput


@register_object("protmol_clip", "model")
class ProteinMoleculeCLIP(AbstractModel):
    def __init__(self, args):
        super(ProteinMoleculeCLIP, self).__init__()
        self.args = args
        self.substrate_encoder = get_object(args.substrate_encoder, "model")(args)
        self.protein_encoder = get_object(args.protein_encoder, "model")(args)
        self.ln_final = nn.LayerNorm(args.chemprop_hidden_dim)  # needs to be shape of protein_hidden, make it chemprop shape since we typically make these match
        self.logit_scale = nn.Parameter(
            torch.ones([]) * torch.log(torch.tensor(1 / 0.07))
        )
        if args.protmol_clip_model_path is not None:
            state_dict = torch.load(args.protmol_clip_model_path)
            state_dict_copy = {
                k.replace("model.", "", 1): v
                for k, v in state_dict["state_dict"].items()
            }
            self.load_state_dict(state_dict_copy)

    def forward(self, batch) -> Dict:
        output = {}
        substrate_features_out = self.substrate_encoder(batch)
        substrate_features = substrate_features_out["hidden"]

        protein_features = self.protein_encoder(
            {"x": batch.sequence, "sequence": batch.sequence, "batch": batch}
        )["hidden"]
        # apply normalization
        protein_features = self.ln_final(protein_features)

        # normalized features
        substrate_features = substrate_features / substrate_features.norm(
            dim=1, keepdim=True
        )
        protein_features = protein_features / protein_features.norm(dim=1, keepdim=True)

        output.update(
            {
                "substrate_features": substrate_features,
                "protein_features": protein_features,
            }
        )
        return output

    @staticmethod
    def add_args(parser) -> None:
        parser.add_argument(
            "--substrate_encoder",
            type=str,
            action=set_nox_type("model"),
            default="chemprop",
            help="Name of encoder to use",
        )
        parser.add_argument(
            "--protein_encoder",
            type=str,
            action=set_nox_type("model"),
            default="fair_esm2",
            help="Name of encoder to use",
        )
        parser.add_argument(
            "--protmol_clip_model_path",
            type=str,
            default=None,
            help="path to saved model if loading from pretrained",
        )


@register_object("protmol_classifier", "model")
class ProtMolClassifier(AbstractModel):
    def __init__(self, args):
        super(ProtMolClassifier, self).__init__()
        self.args = args
        self.enzyme_encoder = get_object(args.enzyme_encoder_name, "model")(args)
        self.substrate_encoder = get_object(args.substrate_encoder_name, "model")(args)
        self.mlp = get_object(args.mlp_name, "model")(args)

    def forward(self, batch):
        # encode molecule
        if self.args.freeze_substrate_encoder:
            self.substrate_encoder.requires_grad_(False)
            with torch.no_grad():
                substrate_dict = self.substrate_encoder(batch)
        else:
            substrate_dict = self.substrate_encoder(batch)
        # encode protein -> must have sequence attribute or key
        x = self.convert_batch_to_seq_list(batch)
        if self.args.freeze_enzyme_encoder:
            self.enzyme_encoder.requires_grad_(False)
            with torch.no_grad():
                enzyme_dict = self.enzyme_encoder(x)
        else:
            enzyme_dict = self.enzyme_encoder(x)

        hidden = torch.cat((enzyme_dict["hidden"], substrate_dict["hidden"]), dim=1)
        mlp_dict = self.mlp({"x": hidden})
        output = {
            "sequence_hidden": enzyme_dict["hidden"],
            "substrate_hidden": substrate_dict["hidden"],
        }
        output.update(mlp_dict)
        return output

    def convert_batch_to_seq_list(self, batch):
        if hasattr(batch, "sequence"):
            x = batch.sequence
        elif "sequence" in batch:
            x = batch["sequence"]
        else:
            raise ValueError("Batch must have sequence attribute or key")
        return x

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--substrate_encoder_name",
            type=str,
            action=set_nox_type("model"),
            default="chemprop",
            help="Name of molecular encoder to use",
        )
        parser.add_argument(
            "--enzyme_encoder_name",
            type=str,
            action=set_nox_type("model"),
            default="non_canon_net",
            help="Name of enzyme encoder to use",
        )
        parser.add_argument(
            "--mlp_name",
            type=str,
            action=set_nox_type("model"),
            default="mlp_classifier",
            help="Name of mlp to use",
        )
        parser.add_argument(
            "--freeze_substrate_encoder",
            action="store_true",
            default=False,
            help="",
        )
        parser.add_argument(
            "--freeze_enzyme_encoder",
            action="store_true",
            default=False,
            help="",
        )


@register_object("protmol_noncanon_classifier", "model")
class ProtMolESMClassifier(ProtMolClassifier):
    def __init__(self, args):
        super(ProtMolESMClassifier, self).__init__(args)
        self.substrate_encoder = self.enzyme_encoder.aa_mol_encoder


@register_object("protmol_clip_classifier", "model")
class ProtMolCLIPClassifier(ProtMolClassifier):
    def __init__(self, args):
        super(ProtMolCLIPClassifier, self).__init__(args)
        # load clip model
        self.clip = get_object(args.enzyme_encoder_name, "model")(args)
        self.enzyme_encoder = self.clip.protein_encoder
        self.substrate_encoder = self.clip.substrate_encoder
