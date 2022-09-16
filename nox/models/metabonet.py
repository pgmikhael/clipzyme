import copy
from typing import Literal
from nox.models.abstract import AbstractModel
from nox.utils.registry import register_object, get_object
from nox.utils.pyg import from_smiles
from nox.utils.classes import set_nox_type
import torch
import torch.nn as nn
import torch.nn.functional as F


@register_object("metabonet_pathways", "model")
class MetaboNetPathways(AbstractModel):
    def __init__(self, args):
        super(MetaboNetPathways, self).__init__()

        self.args = args

        self.metabolite_feature_type = args.metabolite_feature_type
        self.protein_feature_type = args.protein_feature_type

        margs = self.customize_args_for_model(args, "molecule")
        self.molecule_encoder = get_object(args.molecule_model, "model")(margs)

        pargs = self.customize_args_for_model(args, "protein")
        self.protein_encoder = get_object(args.protein_model, "model")(pargs)

        gargs = self.customize_args_for_model(args, "gsm")
        self.gsm_encoder = get_object(args.gsm_model, "model")(gargs)

        self.attention_fc = nn.Linear(args.gsm_node_dim, args.num_pathways)

        args.mlp_input_dim = (
            args.gsm_node_dim + args.gsm_hidden_dim
            if args.concat_drug
            else args.gsm_hidden_dim
        )  # ! num heads?

        self.mlp = get_object(args.final_classifier, "model")(args)

        self.unk_gsm_molecules = args.unk_metabolites
        self.unk_gsm_proteins = args.unk_enzymes

        self.unk_gsm_molecule_embed = torch.nn.Embedding(
            len(args.unk_metabolites), args.gsm_node_dim
        )
        self.unk_gsm_protein_embed = torch.nn.Embedding(
            len(args.unk_enzymes), args.gsm_node_dim
        )

    def customize_args_for_model(
        self, args, input_type: Literal["molecule", "protein", "gsm"]
    ):
        custom_args = copy.deepcopy(args)

        if input_type == "molecule":
            custom_args.linear_input_dim = args.rdkit_features_dim
            custom_args.linear_output_dim = args.gsm_node_dim
            custom_args.gat_node_dim = 9
            custom_args.gat_edge_dim = 3
            custom_args.gat_hidden_dim = (
                args.gsm_node_dim - args.rdkit_features_dim
                if args.use_rdkit_features
                else args.gsm_node_dim
            )
            custom_args.gat_num_heads = args.molecule_model_num_heads
            custom_args.gat_num_layers = args.molecule_model_num_layers  # ! dynamics

        if input_type == "protein":
            custom_args.linear_input_dim = args.protein_dim
            custom_args.linear_output_dim = args.gsm_node_dim

        if input_type == "gsm":
            custom_args.gat_node_dim = args.gsm_node_dim
            custom_args.gat_edge_dim = 3  # !
            custom_args.gat_hidden_dim = args.gsm_hidden_dim
            custom_args.gat_num_heads = args.gsm_num_heads
            custom_args.gat_num_layers = args.gsm_num_layers

        return custom_args

    def forward(self, batch):
        # Encode the molecules
        molecule_embeddings = self.molecule_encoder(batch)

        batch["gsm"].x = self.encode_gsm_entities(batch)

        # Encode the GSMs
        gsm_embeddings = self.gsm_encoder(batch["gsm"])

        # Calculate attention over gsm embeddings
        pathway_attention = F.softmax(
            self.attention_fc(molecule_embeddings), dim=-1
        )  # (batch_size, num_pathways)

        pathways_embeddings = torch.mm(
            batch["gsm"].pathway_mask, gsm_embeddings
        )  # MM[(num_pathways, num_nodes), (num_nodes, hidden)] = (num_pathways, hidden)

        gsm_embedding = torch.mm(
            pathway_attention, pathways_embeddings
        )  # MM[(batch_size, num_pathways), (num_pathways, hidden)] = (batch_size, hidden)

        # Concatenate the embeddings
        if self.args.concat_drug:
            embeddings = torch.cat([molecule_embeddings, gsm_embedding], dim=1)
        else:
            embeddings = gsm_embedding

        # Predict
        return self.mlp(embeddings)

    def encode_gsm_entities(self, batch):
        # use from_smiles & support Nones
        # this may be too slow, might need to do in batches
        node_features = []
        for node in range(batch.data.num_nodes):
            if batch.data.metabolite_features.get(node, False):
                mol = batch.data.metabolite_features[node]
                if self.metabolite_feature_type == "trained":
                    mol = self.molecule_encoder(from_smiles(mol))
                node_features.append(mol)

            elif batch.data.enzyme_features.get(node, False):
                protein = batch.data.enzyme_features[node]
                if self.protein_feature_type in ["precomputed", "trained"]:
                    protein = self.protein_encoder(protein)
                node_features.append(protein)

            elif batch.data.node2type[node] == "metabolite":
                node_features.append(
                    self.unk_gsm_molecule_embed[self.unk_gsm_molecules.index(node)]
                )

            elif batch.data.node2type[node] == "enzyme":
                node_features.append(
                    self.unk_gsm_protein_embed[self.unk_gsm_proteins(node)]
                )

        return torch.stack(node_features)  # (num_nodes, hidden_dim)

    @staticmethod
    def add_args(parser) -> None:
        super(MetaboNetPathways, MetaboNetPathways).add_args(parser)
        parser.add_argument(
            "--protein_model",
            action=set_nox_type("model"),
            type=str,
            default="identity",
            help="name of protein model",
        )
        parser.add_argument(
            "--protein_dim",
            type=int,
            default=1280,
            help="dimensions of protein embedding",
        )
        parser.add_argument(
            "--molecule_model",
            action=set_nox_type("model"),
            type=str,
            default="gatv2",
            help="name of molecule/metabolite model",
        )
        parser.add_argument(
            "--molecule_model_num_heads",
            type=int,
            default=2,
            help="attention heads",
        )
        parser.add_argument(
            "--molecule_num_layers",
            type=int,
            default=2,
            help="num layers",
        )
        parser.add_argument(
            "--gsm_node_dim",
            type=int,
            default=128,
            help="dimensions of final node embedding",
        )
        parser.add_argument(
            "--gsm_model",
            action=set_nox_type("model"),
            type=str,
            default="identity",
            help="name of model for metabolic graph",
        )
        parser.add_argument(
            "--gsm_hidden_dim",
            type=int,
            default=32,
            help="gsm dimension",
        )
        parser.add_argument(
            "--gsm_model_num_heads",
            type=int,
            default=2,
            help="attention heads",
        )
        parser.add_argument(
            "--gsm_num_layers",
            type=int,
            default=2,
            help="num layers",
        )
        parser.add_argument(
            "--final_classifier",
            action=set_nox_type("model"),
            type=str,
            default="mlp_classifier",
            help="name of model for task classifier",
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
        parser.add_argument(
            "--rdkit_features_name",
            type=str,
            default="rdkit_fingerprint",
            help="name of rdkit features to use",
        )
        parser.add_argument(
            "--concat_drug",
            action="store_true",
            default=False,
            help="use drug with pathway representation in final layer",
        )
