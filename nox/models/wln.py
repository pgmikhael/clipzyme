import torch 
import torch.nn as nn
import torch.nn.functional as F
from nox.utils.registry import get_object, register_object
from nox.models.abstract import AbstractModel
from torch_geometric.nn.conv.gatv2_conv import GATv2Conv
from torch_scatter import scatter, scatter_add
from nox.utils.pyg import unbatch
from torch_geometric.utils import to_dense_batch
from nox.models.gat import GAT

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
            default="pairwise_global_attn",
            help="type of global attention to use"
        )


@register_object("reaction_center_net", "model")
class ReactivityCenterNet(AbstractModel):
    def __init__(self, args):
        super().__init__(args)
        self.gat_global_attention = GATWithGlobalAttn(args)
        self.M_a = nn.Linear(args.gat_hidden_dim, args.gat_hidden_dim)
        self.M_b = nn.Linear(args.gat_hidden_dim, args.gat_hidden_dim)
        self.U = nn.Sequential(
            nn.Linear(args.gat_hidden_dim, args.gat_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.gat_hidden_dim, 1) # TODO: Change to predict bond type 
        )

    def forward(self, batch):
        gat_output = self.gat_global_attention(batch) # GAT + Global Attention over node features
        cs = gat_output["node_features"]
        c_tildes = gat_output["node_features_attn"]

        s_uv = self.forward_helper(cs)
        s_uv_tildes = self.forward_helper(c_tildes)

        return {
            "s_uv": s_uv,
            "s_uv_tildes": s_uv_tildes
        }

    def forward_helper(self, node_features, edge_indices, edge_attr, batch_indices):
        # GAT with global attention
        node_features = self.M_a(node_features) # 
        edge_attr = self.M_b(edge_attr)

        # node_features: sparse batch: N x D
        pairwise_node_feats = node_features.unsqueeze(1) + node_features # N x N x D
        # edge_attr: bond features: N x N x D
        s = self.U(edge_attr + pairwise_node_feats).squeeze(-1) # N x N
        out, mask = to_dense_batch(s, batch_indices) # B x N x N
        return out

        # Alternative implementation
        # # Create an empty tensor to hold the scores
        # scores = torch.empty(node_features.shape[0], node_features.shape[0]).to(node_features.device)


        # # For each pair of atoms
        # for i in range(node_features.shape[0]):
        #     for j in range(node_features.shape[0]):
        #         if i != j:
        #             # Get the features for atom pair
        #             atom_pair_features = node_features[i] + node_features[j]
        #             bond_features = graph.edge_attr[graph.edge_index[0] == i, graph.edge_index[1] == j]  # assuming your graph.edge_attr contains bond features

        #             # If there's no direct bond between i and j, bond_features will be an empty tensor
        #             if bond_features.shape[0] == 0:
        #                 continue

        #             # Concatenate the features
        #             # Take the first bond feature if there are multiple bonds between i and j
        #             pair_features = torch.cat((atom_pair_features, global_features[i] + global_features[j], bond_features[0]), dim=0)  
        #             pair_score = self.U(pair_features)

        #             # Add to scores tensor
        #             scores[i, j] = pair_score

        # return scores

@register_object("wldn", "model")
class WLDN(GAT):
    def __init__(self, args):
        super().__init__(args)
        self.reactivity_net = GAT(args) # pretrained reaction center WLN 
        try:
            state_dict = torch.load(args.reactivity_model_path)
            self.reactivity_net.load_state_dict({k[len("model.gat_global_attention."):]: v for k,v in state_dict.items() if k.startswith("model")})
        except:
            print("Could not load pretrained model")
        self.wln_diff = GAT(args) # WLN for difference graph
        self.final_transform = nn.Linear(args.gat_hidden_dim, 1) # for scoring
        
    def forward(self, batch):
        with torch.no_grad():
            reactivity_output = self.reactivity_net(batch)
            reactant_node_feats = reactivity_output["node_features"]
            # product_node_feats = reactivity_output["node_features_attn"] # Peter

        # get candidate products as graph structures
        # product_candidates = self.create_candidates(batch, reactivity_output) # Peter
        # each element in this list is a batch of candidate products (where each batch represents one sample)
        # B x K x N x D (where K is how many candidate products we have for each sample)
        product_candidates_list = self.generate_candidates_from_scores(reactivity_output, batch)

        candidate_scores = []
        for product_candidates in product_candidates_list:
            # get node features for candidate products
            with torch.no_grad():
                candidate_output = self.reactivity_net(product_candidates)
                candidate_node_feats = candidate_output["node_features"]

            # compute difference vectors and replace the node features of the product graph with them
            difference_vectors = candidate_node_feats - reactant_node_feats
            product_candidates.x = difference_vectors

            # apply a separate WLN to the difference graph
            wln_diff_output = self.wln_diff(product_candidates)
            diff_node_feats = wln_diff_output["node_features"]

            # compute the score for each candidate product
            score = self.final_transform(torch.sum(diff_node_feats, dim=-2)) # TODO: figure out dims here and how to return (ie list of lists? Tensor?)
            candidate_scores.append(score)

        return candidate_scores

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