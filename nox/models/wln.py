import torch 
import torch.nn as nn
import torch.nn.functional as F
from nox.utils.registry import get_object, register_object
from nox.utils.classes import set_nox_type
from nox.utils.pyg import unbatch
from nox.utils.wln_processing import generate_candidates_from_scores, get_batch_candidate_bonds
from nox.models.abstract import AbstractModel
from torch_scatter import scatter, scatter_add
from torch_geometric.utils import to_dense_batch, to_dense_adj
from collections import defaultdict
from nox.models.gat import GAT
import copy 
import os 

class WLDN_Cache:
    def __init__(self, path, extension="pt"):
        if not os.path.exists(path):
            os.makedirs(path)
        self.cache_dir = path
        self.files_extension = extension

    def _file_path(self, sample_id):
        return os.path.join(self.cache_dir, f"{sample_id}_candidates.pt")

    def exists(self, sample_id):
        return os.path.isfile(self._file_path(sample_id))

    def get(self, sample_id):
        return torch.load(self._file_path(sample_id))

    def add(self, sample_id, graph):
        torch.save(graph, self._file_path(sample_id))


@register_object("cheap_global_attn", "model")
class CheapGlobalAttention(AbstractModel):
    def __init__(self, args):
        super(CheapGlobalAttention, self).__init__()
        self.linear = nn.Linear(args.gat_hidden_dim, 1)
        
    def forward(self, node_feats, batch_index):
        # node_feats is (N, in_dim)
        scores = self.linear(node_feats)  # (N, 1)
        scores = torch.softmax(scores, dim=0)  # softmax over all nodes
        scores = scores.squeeze(1)  # (N, )
        out = scatter_add(node_feats * scores.unsqueeze(-1), batch_index, dim=0)
        return out

@register_object("pairwise_global_attn", "model")
class PairwiseAttention(AbstractModel):
    def __init__(self, args):
        super(PairwiseAttention, self).__init__()
        self.query_transform = nn.Linear(args.gat_hidden_dim, args.gat_hidden_dim)
        self.key_transform = nn.Linear(args.gat_hidden_dim, args.gat_hidden_dim)

    def forward(self, node_feats, batch_index):
        # Node features: N x F, where N is number of nodes, F is feature dimension
        # Batch index: N, mapping each node to corresponding graph index

        # Compute attention scores
        queries = self.query_transform(node_feats)  # N x F
        keys = self.key_transform(node_feats)  # N x F
        scores = torch.matmul(queries, keys.transpose(-2, -1))  # N x N

        # Mask attention scores to prevent attention to nodes in different graphs
        mask = batch_index[:, None] != batch_index[None, :]  # N x N
        scores = scores.masked_fill(mask, float('-inf'))

        # Compute attention weights
        weights = torch.sigmoid(scores)  # N x N

        # Apply attention weights
        weighted_feats = torch.matmul(weights, node_feats)  # N x F

        return weighted_feats


@register_object("gatv2_globalattn", "model")
class GATWithGlobalAttn(GAT):
    def __init__(self, args):
        super().__init__(args)
        self.global_attention = get_object(args.attn_type, "model")(args)

    def forward(self, graph):
        output = super().forward(graph) # Graph NN (GAT)

        weighted_node_feats = self.global_attention(output["node_features"], graph.batch)  # EQN 6

        output["node_features_attn"] = weighted_node_feats
        return output

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        super(GATWithGlobalAttn, GATWithGlobalAttn).add_args(parser)
        parser.add_argument(
            "--attn_type",
            type=str,
            action=set_nox_type("model"),
            default="pairwise_global_attn",
            help="type of global attention to use"
        )


@register_object("reaction_center_net", "model")
class ReactivityCenterNet(AbstractModel):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.gat_global_attention = get_object(args.gat_type, "model")(args) # GATWithGlobalAttn(args)
        self.M_a = nn.Linear(args.gat_hidden_dim, args.gat_hidden_dim)
        self.M_b = nn.Linear(args.gat_edge_dim, args.gat_hidden_dim)
        self.lin = nn.Linear(2*args.gat_hidden_dim, args.gat_hidden_dim)
        self.U = nn.Sequential(
            nn.Linear(args.gat_hidden_dim, args.gat_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.gat_hidden_dim, args.num_predicted_bond_types) # TODO: Change to predict bond type 
        )

    def forward(self, batch):
        gat_output = self.gat_global_attention(batch['reactants']) # GAT + Global Attention over node features
        cs = gat_output["node_features"]
        c_tildes = gat_output["node_features_attn"]
        c_final = self.lin(torch.cat([cs, c_tildes], dim=-1)) # N x 2*hidden_dim -> N x hidden_dim

        s_uv = self.forward_helper(c_final, batch['reactants']['edge_index'], batch['reactants']['edge_attr'], batch['reactants']['batch'])

        # precompute for top k metric
        candidate_bond_changes = get_batch_candidate_bonds(batch["reaction"], s_uv.detach(), batch['reactants'].batch)
        # make bonds that are "4" -> "1.5"
        for i in range(len(candidate_bond_changes)):
            candidate_bond_changes[i] = [(elem[0], elem[1], 1.5, elem[3]) if elem[2] == 4 else elem for elem in candidate_bond_changes[i]]

        batch_real_bond_changes = []
        for i in range(len(batch['reactants']['bond_changes'])):
            reaction_real_bond_changes = []
            for elem in batch['reactants']['bond_changes'][i]:
                reaction_real_bond_changes.append(tuple(elem))
            batch_real_bond_changes.append(reaction_real_bond_changes)

        assert len(candidate_bond_changes) == len(batch_real_bond_changes)

        return {
            "s_uv": s_uv,
            "candidate_bond_changes": candidate_bond_changes,
            "real_bond_changes": batch_real_bond_changes
        }

    def forward_helper(self, node_features, edge_indices, edge_attr, batch_indices):
        # GAT with global attention
        node_features = self.M_a(node_features) # N x hidden_dim -> N x hidden_dim 
        edge_attr = self.M_b(edge_attr.float()) # E x 3 -> E x hidden_dim 

        # convert to dense adj: E x hidden_dim -> N x N x hidden_dim
        dense_edge_attr = to_dense_adj(edge_index = edge_indices, edge_attr = edge_attr).squeeze(0)

        # node_features: sparse batch: N x D
        pairwise_node_feats = node_features.unsqueeze(1) + node_features # N x N x D
        # edge_attr: bond features: N x N x D
        s = self.U(dense_edge_attr + pairwise_node_feats).squeeze(-1) # N x N
        # removed this line since the sizes become inconsistent later
        # s, mask = to_dense_batch(s, batch_indices) # B x max_batch_N x N x num_predicted_bond_types
        
        # make symmetric
        s = (s + s.transpose(0,1))/2
        return s

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        super(ReactivityCenterNet, ReactivityCenterNet).add_args(parser)
        parser.add_argument(
            "--num_predicted_bond_types",
            type=int,
            default=5,
            help="number of bond types to predict, this is t in the paper"
        )
        parser.add_argument(
            "--gat_type",
            type=str,
            action=set_nox_type("model"),
            default="gatv2_globalattn",
            help="Type of gat to use, mainly to init args"
        )
        parser.add_argument(
            "--topk_bonds",
            nargs='+',
            type=int,
            default=[1, 3, 5],
            help="topk bonds to consider for accuracy metric"
        )


@register_object("wldn", "model")
class WLDN(GATWithGlobalAttn):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.reactivity_net = get_object(args.reactivity_net_type, "model")(args)
        try:
            state_dict = torch.load(args.reactivity_model_path)
            self.reactivity_net.load_state_dict({k[len("model.gat_global_attention."):]: v for k,v in state_dict.items() if k.startswith("model")})
        except:
            print("Could not load pretrained model")
        self.wln = GAT(args) # WLN for mol representation
        wln_diff_args = copy.deepcopy(args)
        wln_diff_args.gat_node_dim = args.gat_hidden_dim
        # wln_diff_args.gat_edge_dim = args.gat_hidden_dim
        self.wln_diff = GAT(wln_diff_args) # WLN for difference graph
        self.final_transform = nn.Linear(args.gat_hidden_dim, 1) # for scoring
        self.use_cache = args.cache_path is not None 
        if self.use_cache:
            self.cache = WLDN_Cache(os.path.join(args.cache_path, args.experiment_name), "pt")

    def get_reactivity_scores_and_candidates(self, batch):
        """Runs through GNN to obtain reactivity scores then uses them to generate product"""

        with torch.no_grad():
            reactivity_output = self.reactivity_net(batch)
        reactant_node_feats = self.wln(batch["reactants"])["node_features"] # N x D, where N is all the nodes in the batch
        dense_reactant_node_feats, mask = to_dense_batch(reactant_node_feats, batch=batch["reactants"].batch) # B x max_batch_N x D

        # get candidate products as graph structures
        # each element in this list is a batch of candidate products (where each batch represents one sample)
        if self.training:
            mode = "train"
        else:
            mode = "test"
        
        product_candidates_list = generate_candidates_from_scores(reactivity_output, batch, self.args, mode)
        return product_candidates_list

    def forward(self, batch):
        if self.use_cache:
            if not all( self.cache.exists(sid) for sid in batch["sample_id"] ):
                product_candidates_list = self.get_reactivity_scores_and_candidates(batch)
                [self.cache.add(sid, product_candidates) for sid, product_candidates in zip(batch["sample_id"], product_candidates_list)]
            else:
                product_candidates_list =  [self.cache.get(sid) for sid in batch["sample_id"]]
        else:
            product_candidates_list = self.get_reactivity_scores_and_candidates(batch)

        candidate_scores = []
        for idx, product_candidates in enumerate(product_candidates_list):
            # get node features for candidate products
            product_candidates = product_candidates.to(reactant_node_feats.device)
            candidate_node_feats = self.wln(product_candidates)["node_features"]
            dense_candidate_node_feats, mask = to_dense_batch(candidate_node_feats, batch=product_candidates.batch) # B x num_nodes x D
            
            num_nodes = dense_candidate_node_feats.shape[1]

            # compute difference vectors and replace the node features of the product graph with them
            difference_vectors = dense_candidate_node_feats - dense_reactant_node_feats[idx][:num_nodes].unsqueeze(0)

            # undensify
            total_nodes = dense_candidate_node_feats.shape[0] * num_nodes
            difference_vectors = difference_vectors.view(total_nodes, -1)
            product_candidates.x = difference_vectors
            
            # apply a separate WLN to the difference graph
            wln_diff_output = self.wln_diff(product_candidates)
            diff_node_feats = wln_diff_output["node_features"]

            # compute the score for each candidate product
            # to dense
            diff_node_feats, _ = to_dense_batch(diff_node_feats, product_candidates.batch)
            score = self.final_transform(torch.sum(diff_node_feats, dim=-2))
            candidate_scores.append(score) # K x 1

        # ? can k may be different per sample?
        # candidates_scores = torch.cat(candidate_scores, dim=0) # B x K x 1

        # note: dgl implementation adds reactivity score
        output = {
            "logit": candidate_scores,
            "product_candidates_list": product_candidates_list,
            "s_uv": reactivity_output["s_uv"], # for debugging purposes
            }
        return output

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        super(WLDN, WLDN).add_args(parser)
        parser.add_argument(
            "--num_candidate_bond_changes",
            type=int,
            default=20,
            help="Core size"
        )
        parser.add_argument(
            "--max_num_bond_changes",
            type=int,
            default=5,
            help="Combinations"
        )
        parser.add_argument(
            "--max_num_change_combos_per_reaction",
            type=int,
            default=500,
            help="cutoff"
        )
        parser.add_argument(
            "--reactivity_net_type",
            type=str,
            action=set_nox_type("model"),
            default="reaction_center_net",
            help="Type of reactivity net to use, mainly to init args"
        )