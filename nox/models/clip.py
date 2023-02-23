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

        self.substrate_encoder = get_object(args.substrate_encoder, "model")(args)
        self.protein_encoder = get_object(args.protein_encoder, "model")(args)
        self.ln_final = nn.LayerNorm(args.num_classes)
        self.logit_scale = nn.Parameter(torch.ones([]) *  torch.log(torch.tensor(1 / 0.07)))

    def forward(self, batch) -> Dict:
        output = {}
        substrate_features_out = self.substrate_encoder(batch)
        substrate_features = substrate_features_out["hidden"]
 
        protein_features = self.protein_encoder(batch)["hidden"]
        # apply normalization
        protein_features = self.ln_final(protein_features)

        # normalized features
        substrate_features = substrate_features / substrate_features.norm(dim=1, keepdim=True)
        protein_features = protein_features / protein_features.norm(dim=1, keepdim=True)

        output.update( {"substrate_features": substrate_features, "protein_features": protein_features})
        return output

    @staticmethod
    def add_args(parser) -> None:
        parser.add_argument(
            "--substrate_encoder",
            type=str,
            action=set_nox_type("model"),
            default="graph_classifier",
            help="Name of encoder to use",
        )
        parser.add_argument(
            "--protein_encoder",
            type=str,
            action=set_nox_type("model"),
            default="protein_encoder",
            help="Name of encoder to use",
        )