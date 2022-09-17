import copy
from typing import Literal
from nox.models.abstract import AbstractModel
from nox.utils.registry import register_object, get_object
from nox.utils.pyg import from_smiles
from nox.utils.classes import set_nox_type
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch


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
        )  

        self.mlp = get_object(args.final_classifier, "model")(args)

        # init embedding table for entities with unknown structure
        self.unk_gsm_molecules = args.unk_metabolites
        self.unk_gsm_proteins = args.unk_enzymes

        if len(self.unk_gsm_molecules) != 0:
            self.unk_gsm_molecule_embed = torch.nn.Embedding(
                len(args.unk_metabolites), args.gsm_node_dim
            )

        if len(self.unk_gsm_proteins) != 0:
            self.unk_gsm_protein_embed = torch.nn.Embedding(
                len(args.unk_enzymes), args.gsm_node_dim
            )
        
        if self.transform_mol_dim:
            self.mol_to_gsm_dim_fc = nn.Linear(args.gsm_node_dim + args.rdkit_features_dim, args.gsm_node_dim)

        if self.transform_protein_dim:
            self.prot_to_gsm_dim_fc = nn.Linear(args.protein_dim, args.gsm_node_dim)

    def customize_args_for_model(
        self, args, input_type: Literal["molecule", "protein", "gsm"]
    ):
        custom_args = copy.deepcopy(args)

        if input_type == "molecule":
            custom_args.linear_input_dim = args.rdkit_features_dim
            custom_args.linear_output_dim = args.gsm_node_dim
            custom_args.gat_node_dim = 9
            custom_args.gat_edge_dim = 3
            custom_args.gat_hidden_dim = args.gsm_node_dim
            custom_args.gat_num_heads = args.molecule_num_heads
            custom_args.gat_num_layers = args.molecule_num_layers 
            self.transform_mol_dim =  (args.use_rdkit_features) and (args.metabolite_feature_type == 'trained')
                
        if input_type == "protein":
            custom_args.linear_input_dim = args.protein_dim
            custom_args.linear_output_dim = args.gsm_node_dim
            custom_args.num_embed = args.num_proteins
            custom_args.embed_dim = args.gsm_node_dim
            self.transform_protein_dim = (args.protein_dim != args.gsm_node_dim) and (args.protein_feature_type == 'trained')

        if input_type == "gsm":
            custom_args.gat_node_dim = args.gsm_node_dim
            custom_args.gat_edge_dim = 1  
            custom_args.gat_hidden_dim = args.gsm_hidden_dim
            custom_args.gat_num_heads = args.gsm_num_heads
            custom_args.gat_num_layers = args.gsm_num_layers
            custom_args.gat_pool = "none"

        return custom_args

    def forward(self, batch):
        # Encode the molecules
        batch_size = batch["mol"].batch.unique().shape[0]
        if self.metabolite_feature_type == "precomputed":
            mol_features = batch["mol"]["rdkit_features"].view(batch_size, -1).float()
            molecule_embeddings = self.molecule_encoder(mol_features)["hidden"]  
        else:
            molecule_embeddings = self.molecule_encoder(batch["mol"])["hidden"]
            if self.args.use_rdkit_features:
                features = batch["mol"]["rdkit_features"].view(batch_size, -1).float()
                molecule_embeddings = torch.concat([molecule_embeddings, features ], dim=-1)
        
        # project to gsm node dim if trained and fp 
        if self.transform_mol_dim:
            molecule_embeddings = self.mol_to_gsm_dim_fc(molecule_embeddings)

        batch["gsm"].x = self.encode_gsm_entities(batch["gsm"])

        # Encode the GSMs
        gsm_embeddings = self.gsm_encoder(batch["gsm"])["node_features"]

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
        return self.mlp({"x": embeddings})

    def encode_gsm_entities(self, gsm):
        # use from_smiles & support Nones
        # this may be too slow, might need to do in batches
        node_features = []
        metabolites = []
        proteins = []

        for node in range(gsm.num_nodes):
            if gsm.metabolite_features.get(node, None) is not None:
                mol = gsm.metabolite_features[node]
                metabolites.append((node, mol))

            elif gsm.enzyme_features.get(node, None) is not None:
                protein = gsm.enzyme_features[node]
                proteins.append((node, protein))

        if len(self.unk_gsm_molecules) > 0:
            node_features += list(zip(self.unk_gsm_molecules, self.unk_gsm_molecule_embed.weight))
        
        if len(self.unk_gsm_proteins) > 0:
            node_features += list(zip(self.unk_gsm_proteins, self.unk_gsm_protein_embed.weight))

        follow_batch = None
        exclude_keys = None
        for i in range(0, len(metabolites), self.args.batch_size):
            indx_batch = [j[0] for j in metabolites[i:i+self.args.batch_size]]
            met_batch = [j[1] for j in metabolites[i:i+self.args.batch_size]]

            if self.args.metabolite_feature_type == "precomputed":
                mol_batch = torch.stack(met_batch)
            
            elif self.args.metabolite_feature_type == "trained":
                mol_batch = Batch.from_data_list(met_batch, follow_batch, exclude_keys) 
            
            mol_embeds = self.molecule_encoder(mol_batch)['hidden']
            if self.args.use_rdkit_features and self.args.metabolite_feature_type == "trained":
                features = mol_batch["rdkit_features"].view(len(met_batch), -1).float()
                mol_embeds = torch.concat([mol_embeds, features ], dim=-1)
            
            # project to gsm node dim if trained and fp 
            if self.transform_mol_dim:
                mol_embeds = self.mol_to_gsm_dim_fc(mol_embeds)

            indexed_mol_embeds = list(zip(indx_batch, mol_embeds))
            node_features += indexed_mol_embeds

        for i in range(0, len(proteins), self.args.batch_size):
            indx_batch = [j[0] for j in proteins[i:i+self.args.batch_size]]
            prot_batch = [j[1] for j in proteins[i:i+self.args.batch_size]]
            if self.args.protein_feature_type == "precomputed":
                prot_batch = torch.stack(prot_batch)
            
            prot_embeds = self.protein_encoder(prot_batch)['hidden']
            # project to gsm node dim if trained 
            if self.transform_protein_dim:
                prot_embeds = self.prot_to_gsm_dim_fc(prot_embeds)

            indexed_prot_embeds = list(zip(indx_batch, prot_embeds))
            node_features += indexed_prot_embeds

        # sort node features
        node_features = [feat[1] for feat in sorted(node_features, key=lambda x: x[0])]

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
            "--molecule_num_heads",
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
            "--gsm_num_heads",
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
