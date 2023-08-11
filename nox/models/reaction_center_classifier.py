import torch 
import torch.nn as nn
import torch.nn.functional as F
from nox.utils.registry import get_object, register_object
from nox.utils.classes import set_nox_type
from nox.utils.pyg import unbatch
from nox.models.abstract import AbstractModel
from torch_scatter import scatter, scatter_add
from torch_geometric.utils import to_dense_batch, to_dense_adj, dense_to_sparse
from collections import defaultdict
from rdkit import Chem 
import copy 
import os 
from rich import print

@register_object("pretrained_reaction_center_classifier", "model")
class ReactionCenterClassifier(AbstractModel):
    def __init__(self, args):
        super().__init__()
        self.args = args
        try:
            state_dict = torch.load(args.reactivity_model_path)
            self.reactivity_net = get_object(args.reactivity_net_type, "model")(state_dict['hyper_parameters']['args'])
            self.reactivity_net.load_state_dict({k[len("model."):]: v for k,v in state_dict["state_dict"].items() if k.startswith("model")})
            self.reactivity_net.requires_grad_(False)
            print(f"[bold] Loaded checkpoint from {args.reactivity_model_path}")
        except:
            self.reactivity_net = get_object(args.reactivity_net_type, "model")(args).requires_grad_(False)
            print("Could not load pretrained model")
        
        self.freeze_encoder = args.freeze_reaction_center
        if self.freeze_encoder:
            self.reactivity_net.requires_grad_(False)

        self.classifier = get_object(args.classifier, "model")(args)
        

    def forward(self, batch):
        if self.freeze_encoder:
            self.reactivity_net.requires_grad_(False)
            with torch.no_grad():
                reactivity_output = self.reactivity_net(batch)
        else:
            reactivity_output = self.reactivity_net(batch)
        
        hidden = reactivity_output["c_final"]
        if getattr(self.args, "project_nodes", False):
            hidden = self.reactivity_net.M_a(hidden)

        hidden = scatter(hidden, batch['reactants'].batch, dim=0, reduce="mean")
        if self.args.use_rdkit_features:
            hidden = torch.hstack([hidden, batch['reactants'].rdkit_features]).float()
        outputs = self.classifier({"x":hidden})
        return outputs 

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        super(ReactionCenterClassifier, ReactionCenterClassifier).add_args(parser)
        parser.add_argument(
            "--reactivity_net_type",
            type=str,
            action=set_nox_type("model"),
            default="reaction_center_net",
            help="Type of reactivity net to use, mainly to init args"
        )
        parser.add_argument(
            "--reactivity_model_path",
            type=str,
            help="path to pretrained reaction center prediction model"
        )
        parser.add_argument(
            "--classifier",
            type=str,
            action=set_nox_type("model"),
            default="mlp_classifier",
            help="mlp"
        )
        parser.add_argument(
            "--freeze_reaction_center",
            action="store_true",
            default=False,
            help="whether use model as pre-trained encoder and not update weights",
        )
        parser.add_argument(
            "--project_nodes",
            action="store_true",
            default=False,
            help="use Ma in model",
        )
        parser.add_argument(
            "--use_rdkit_features",
            action="store_true",
            default=False,
            help="whether using graph-level features from rdkit",
        )
        parser.add_argument(
            "--rdkit_features_dim",
            type=int,
            default=0,
            help="number of features",
        )
        