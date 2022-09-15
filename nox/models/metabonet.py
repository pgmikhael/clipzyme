import copy
from nox.models.abstract import AbstractModel
from nox.models.classifier import MLPClassifier, GraphClassifier
from nox.utils.registry import register_object, get_object
from nox.utils.pyg import from_smiles
from nox.utils.classes import set_nox_type
import torch
import torch.nn.functional as F
from nox.models.gat import GAT


@register_object("metabonet_pathways", "model")
class MetaboNetPathways(AbstractModel):
    def __init__(self, args):
        super(MetaboNetPathways, self).__init__()

        self.args = args

        self.metabolite_feature_type = args.metabolite_feature_type
        self.protein_feature_type = args.protein_feature_type

        margs = copy.deepcopy(args)
        margs.linear_input_dim = args.metabolite_dim
        margs.linear_output_dim = args.node_dim
        self.metabolite_encoder = get_object(args.metabolite_model, "model")(margs)
        
        pargs = copy.deepcopy(args)
        pargs.linear_input_dim = args.protein_dim
        pargs.linear_output_dim = args.node_dim
        self.protein_encoder = get_object(args.protein_model, "model")(pargs)

        self.gsm_encoder = get_object(args.gsm_model, "model")(args)

        self.mlp = get_object(args.final_classifier, "model")(args)

        # !
        self.unk_gsm_molecules = args.unk_metabolites
        self.unk_gsm_proteins = args.unk_enzymes

        self.unk_gsm_molecule_embed = torch.nn.Embedding(
            len(args.unk_metabolites), args.node_out_dim
        )
        self.unk_gsm_protein_embed = torch.nn.Embedding(
            len(args.unk_enzymes), args.node_out_dim
        )

    def forward(self, batch):
        # Encode the molecules
        molecule_embeddings = self.molecule_encoder(batch)

        batch["gsm"].x = self.encode_gsm_entities(batch)

        # Encode the GSMs
        gsm_embeddings = self.gsm_encoder(batch["gsm"])

        # Calculate attention over gsm embeddings
        # ! not how attention works
        attention = F.softmax(molecule_embeddings, dim=1)  # (batch_size, num_pathways)

        pathways_embeddings = torch.mm(
            batch["gsm"].pathway_mask, gsm_embeddings
        )  # MM[(num_pathways, num_nodes), (num_nodes, hidden)] = (num_pathways, hidden)

        gsm_embedding = torch.mm(
            attention, pathways_embeddings
        )  # MM[(batch_size, num_pathways), (num_pathways, hidden)] = (batch_size, hidden)

        # Concatenate the embeddings
        embeddings = torch.cat([molecule_embeddings, gsm_embedding], dim=1)

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
            "--metabolite_model",
            action=set_nox_type("model"),
            type=str,
            default="gatv2",
            help="name of molecule/metabolite model",
        )
        parser.add_argument(
            "--metabolite_dim",
            type=int,
            default=2048,
            help="dimensions of metabolite embedding",
        )
        parser.add_argument(
            "--node_dim",
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
