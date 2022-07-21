import copy
from collections.abc import Sequence

import torch
from torch import nn, autograd
from torch_scatter import scatter_add
from torch_geometric.data import Batch

from nox.utils.loading import default_collate
from nox.utils.registry import get_object, register_object
from nox.utils.nbf import gen_rel_conv_layer, nbf_utils
from nox.utils.classes import set_nox_type
from nox.models.abstract import AbstractModel


@register_object("nbfnet", "model")
class NBFNet(AbstractModel):
    def __init__(self, args) -> None:
        """
        Initialize a Neural-Bellman Ford Model
        Code Adapted from: https://github.com/KiddoZhu/NBFNet-PyG
        """
        super(NBFNet, self).__init__()
        self.num_negative = args.num_negative
        self.strict_negative = args.strict_negative

        if not isinstance(args.hidden_dims, Sequence):
            hidden_dims = [args.hidden_dims]
        else:
            hidden_dims = args.hidden_dims

        self.dims = [args.input_dim] + list(hidden_dims)
        self.num_relation = args.num_relations
        self.short_cut = (
            args.short_cut  # whether to use residual connections between GNN layers
        )
        self.concat_hidden = (
            args.concat_hidden
        )  # whether to compute final states as a function of all layer outputs or last
        self.remove_one_hop = (
            args.remove_one_hop
        )  # whether to dynamically remove one-hop edges from edge_index
        self.num_beam = args.num_beam
        self.path_topk = args.path_topk

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                gen_rel_conv_layer.GeneralizedRelationalConv(
                    self.dims[i],
                    self.dims[i + 1],
                    args.num_relations,
                    self.dims[0],
                    args.message_func,
                    args.aggregate_func,
                    args.layer_norm,
                    args.activation_func,
                    args.dependent,
                )
            )

        feature_dim = (
            sum(hidden_dims) if args.concat_hidden else hidden_dims[-1]
        ) + args.input_dim

        # additional relation embedding which serves as an initial 'query' for the NBFNet forward pass
        # each layer has its own learnable relations matrix, so we send the total number of relations, too
        self.query = nn.Embedding(args.num_relations, args.input_dim)
        self.mlp = nn.Sequential()
        mlp = []
        for i in range(args.num_mlp_layer - 1):
            mlp.append(nn.Linear(feature_dim, feature_dim))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(feature_dim, 1))
        self.mlp = nn.Sequential(*mlp)

    def remove_easy_edges(self, data, h_index, t_index, r_index=None):
        # we remove training edges (we need to predict them at training time) from the edge index
        # think of it as a dynamic edge dropout
        h_index_ext = torch.cat([h_index, t_index], dim=-1)
        t_index_ext = torch.cat([t_index, h_index], dim=-1)
        r_index_ext = torch.cat([r_index, r_index + self.num_relation // 2], dim=-1)
        if self.remove_one_hop:
            # we remove all existing immediate edges between heads and tails in the batch
            edge_index = data.edge_index
            easy_edge = torch.stack([h_index_ext, t_index_ext]).flatten(1)
            index = nbf_utils.edge_match(edge_index, easy_edge)[0]
            mask = ~nbf_utils.index_to_mask(index, data.num_edges)
        else:
            # we remove existing immediate edges between heads and tails in the batch with the given relation
            edge_index = torch.cat([data.edge_index, data.edge_type.unsqueeze(0)])
            # note that here we add relation types r_index_ext to the matching query
            easy_edge = torch.stack([h_index_ext, t_index_ext, r_index_ext]).flatten(1)
            index = nbf_utils.edge_match(edge_index, easy_edge)[0]
            mask = ~nbf_utils.index_to_mask(index, data.num_edges)

        data = copy.copy(data)
        data.edge_index = data.edge_index[:, mask]
        data.edge_type = data.edge_type[mask]
        return data

    def negative_sample_to_tail(self, h_index, t_index, r_index):
        # convert p(h | t, r) to p(t' | h', r')
        # h' = t, r' = r^{-1}, t' = h
        is_t_neg = (h_index == h_index[:, [0]]).all(dim=-1, keepdim=True)
        new_h_index = torch.where(is_t_neg, h_index, t_index)
        new_t_index = torch.where(is_t_neg, t_index, h_index)
        new_r_index = torch.where(is_t_neg, r_index, r_index + self.num_relation // 2)
        return new_h_index, new_t_index, new_r_index

    def init_boundary_conditions(self, data, h_index, r_index):
        batch_size = len(r_index)

        # initialize queries (relation types of the given triples)
        query = self.query(r_index)
        index = h_index.unsqueeze(-1).expand_as(query)

        boundary = torch.zeros(
            batch_size, data.num_nodes, self.dims[0], device=h_index.device
        )
        # by the scatter operation we put query (relation) embeddings as init features of source (index) nodes
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))

        return query, boundary

    def bellmanford(self, data, h_index, r_index, separate_grad=False):

        query, boundary = self.init_boundary_conditions(data, h_index, r_index)

        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=h_index.device)

        hiddens = []
        edge_weights = []
        layer_input = boundary

        for layer in self.layers:
            if separate_grad:
                edge_weight = edge_weight.clone().requires_grad_()
            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            hidden = layer(
                layer_input,
                query,
                boundary,
                data.edge_index,
                data.edge_type,
                size,
                edge_weight,
            )
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(
            -1, data.num_nodes, -1
        )  # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, batch):
        # get full graph data
        data = batch["graph"]

        # sample negatives
        batch_with_negatives = batch["triplet"]

        h_index, t_index, r_index = batch_with_negatives.unbind(-1)
        if self.training:
            # Edge dropout in the training mode
            # here we want to remove immediate edges (head, relation, tail) from the edge_index and edge_types
            # to make NBFNet iteration learn non-trivial paths
            data = self.remove_easy_edges(data, h_index, t_index, r_index)

        shape = h_index.shape
        # turn all triples in a batch into a tail prediction mode
        h_index, t_index, r_index = self.negative_sample_to_tail(
            h_index, t_index, r_index
        )
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()

        # message passing and updated node representations
        output = self.bellmanford(
            data, h_index[:, 0], r_index[:, 0]
        )  # (num_nodes, batch_size, feature_dim）
        feature = output["node_feature"]
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        # extract representations of tail entities from the updated node states
        feature = feature.gather(
            1, index
        )  # (batch_size, num_negative + 1, feature_dim)

        # probability logit for each tail node in the batch
        # (batch_size, num_negative + 1, dim) -> (batch_size, num_negative + 1)
        score = self.mlp(feature).squeeze(-1)

        output = {"logit": score.view(shape)}
        return output

    def visualize(self, data, batch):
        assert batch.shape == (1, 3)
        h_index, t_index, r_index = batch.unbind(-1)

        output = self.bellmanford(data, h_index, r_index, separate_grad=True)
        feature = output["node_feature"]
        edge_weights = output["edge_weights"]

        index = t_index.unsqueeze(0).unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        feature = feature.gather(1, index).squeeze(0)
        score = self.mlp(feature).squeeze(-1)

        edge_grads = autograd.grad(score, edge_weights)
        distances, back_edges = self.beam_search_distance(
            data, edge_grads, h_index, t_index, self.num_beam
        )
        paths, weights = self.topk_average_length(
            distances, back_edges, t_index, self.path_topk
        )

        return paths, weights

    @torch.no_grad()
    def beam_search_distance(self, data, edge_grads, h_index, t_index, num_beam=10):
        # beam search the top-k distance from h to t (and to every other node)
        num_nodes = data.num_nodes
        input = torch.full((num_nodes, num_beam), float("-inf"), device=h_index.device)
        input[h_index, 0] = 0
        edge_mask = data.edge_index[0, :] != t_index

        distances = []
        back_edges = []
        for edge_grad in edge_grads:
            # we don't allow any path goes out of t once it arrives at t
            node_in, node_out = data.edge_index[:, edge_mask]
            relation = data.edge_type[edge_mask]
            edge_grad = edge_grad[edge_mask]

            message = input[node_in] + edge_grad.unsqueeze(-1)  # (num_edges, num_beam)
            # (num_edges, num_beam, 3)
            msg_source = (
                torch.stack([node_in, node_out, relation], dim=-1)
                .unsqueeze(1)
                .expand(-1, num_beam, -1)
            )

            # (num_edges, num_beam)
            is_duplicate = torch.isclose(
                message.unsqueeze(-1), message.unsqueeze(-2)
            ) & (msg_source.unsqueeze(-2) == msg_source.unsqueeze(-3)).all(dim=-1)
            # pick the first occurrence as the ranking in the previous node's beam
            # this makes deduplication easier later
            # and store it in msg_source
            is_duplicate = is_duplicate.float() - torch.arange(
                num_beam, dtype=torch.float, device=message.device
            ) / (num_beam + 1)
            prev_rank = is_duplicate.argmax(dim=-1, keepdim=True)
            msg_source = torch.cat(
                [msg_source, prev_rank], dim=-1
            )  # (num_edges, num_beam, 4)

            node_out, order = node_out.sort()
            node_out_set = torch.unique(node_out)
            # sort messages w.r.t. node_out
            message = message[order].flatten()  # (num_edges * num_beam)
            msg_source = msg_source[order].flatten(0, -2)  # (num_edges * num_beam, 4)
            size = node_out.bincount(minlength=num_nodes)
            msg2out = nbf_utils.size_to_index(size[node_out_set] * num_beam)
            # deduplicate messages that are from the same source and the same beam
            is_duplicate = (msg_source[1:] == msg_source[:-1]).all(dim=-1)
            is_duplicate = torch.cat(
                [torch.zeros(1, dtype=torch.bool, device=message.device), is_duplicate]
            )
            message = message[~is_duplicate]
            msg_source = msg_source[~is_duplicate]
            msg2out = msg2out[~is_duplicate]
            size = msg2out.bincount(minlength=len(node_out_set))

            if not torch.isinf(message).all():
                # take the topk messages from the neighborhood
                # distance: (len(node_out_set) * num_beam)
                distance, rel_index = nbf_utils.scatter_topk(message, size, k=num_beam)
                abs_index = rel_index + (size.cumsum(0) - size).unsqueeze(-1)
                # store msg_source for backtracking
                back_edge = msg_source[abs_index]  # (len(node_out_set) * num_beam, 4)
                distance = distance.view(len(node_out_set), num_beam)
                back_edge = back_edge.view(len(node_out_set), num_beam, 4)
                # scatter distance / back_edge back to all nodes
                distance = scatter_add(
                    distance, node_out_set, dim=0, dim_size=num_nodes
                )  # (num_nodes, num_beam)
                back_edge = scatter_add(
                    back_edge, node_out_set, dim=0, dim_size=num_nodes
                )  # (num_nodes, num_beam, 4)
            else:
                distance = torch.full(
                    (num_nodes, num_beam), float("-inf"), device=message.device
                )
                back_edge = torch.zeros(
                    num_nodes, num_beam, 4, dtype=torch.long, device=message.device
                )

            distances.append(distance)
            back_edges.append(back_edge)
            input = distance

        return distances, back_edges

    def topk_average_length(self, distances, back_edges, t_index, k=10):
        # backtrack distances and back_edges to generate the paths
        paths = []
        average_lengths = []

        for i in range(len(distances)):
            distance, order = distances[i][t_index].flatten(0, -1).sort(descending=True)
            back_edge = back_edges[i][t_index].flatten(0, -2)[order]
            for d, (h, t, r, prev_rank) in zip(
                distance[:k].tolist(), back_edge[:k].tolist()
            ):
                if d == float("-inf"):
                    break
                path = [(h, t, r)]
                for j in range(i - 1, -1, -1):
                    h, t, r, prev_rank = back_edges[j][h, prev_rank].tolist()
                    path.append((h, t, r))
                paths.append(path[::-1])
                average_lengths.append(d / len(path))

        if paths:
            average_lengths, paths = zip(
                *sorted(zip(average_lengths, paths), reverse=True)[:k]
            )

        return paths, average_lengths

    @staticmethod
    def add_args(parser) -> None:
        parser.add_argument(
            "--input_dim", type=int, default=32, help="size of input features"
        )
        parser.add_argument(
            "--hidden_dims",
            type=int,
            nargs="*",
            default=[32, 32, 32, 32, 32, 32],
            help="size of hidden dimensions features",
        )
        parser.add_argument(
            "--num_relations",
            type=int,
            default=None,
            help="number of relationships set by the dataset",
        )
        parser.add_argument(
            "--message_func",
            type=str,
            choices=["transe", "distmult", "rotate"],
            default="distmult",
            help="name of message function. one of [transe, distmult, rotate]",
        )
        parser.add_argument(
            "--aggregate_func",
            type=str,
            choices=["sum", "mean", "max", "pna"],
            default="pna",
            help="name of message function. one of [sum, mean, max, pna]",
        )
        parser.add_argument(
            "--short_cut",
            action="store_true",
            default=False,
            help="whether to use skip connections",
        )
        parser.add_argument(
            "--layer_norm",
            action="store_true",
            default=False,
            help="whether to user layer normalization",
        )
        parser.add_argument(
            "--activation_func",
            type=str,
            default="relu",
            help="name of torch.nn.functional activation function",
        )
        parser.add_argument(
            "--concat_hidden",
            action="store_true",
            default=False,
            help="whether to concatenate hiddens of different layers",
        )
        parser.add_argument(
            "--num_mlp_layer",
            type=int,
            default=2,
            help="number of MLP layers",
        )
        parser.add_argument(
            "--dependent",
            action="store_true",
            default=False,
            help="whether to make relation embedding a projection of existing query or a learned embedding",
        )
        parser.add_argument(
            "--remove_one_hop",
            action="store_true",
            default=False,
            help="whether to remove all existing immediate edges between heads and tails in the batch",
        )
        parser.add_argument(
            "--num_beam",
            type=int,
            default=10,
            help="beam search the top-k distance from h to t (and to every other node)",
        )
        parser.add_argument(
            "--path_topk",
            type=int,
            default=10,
            help="number of paths to use to obtain average length of the top-k paths",
        )
        parser.add_argument(
            "--num_negative",
            type=int,
            default=32,
            help="number of negative samples to use",
        )
        parser.add_argument(
            "--strict_negative",
            action="store_true",
            default=False,
            help="whether to only consider samples with known no edges as negative examples",
        )


@register_object("metabo_nbfnet", "model")
class Metabo_NBFNet(NBFNet):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.protein_encoder = get_object(args.protein_model, "model")(args)
        self.metabolite_encoder = get_object(args.metabolite_model, "model")(args)

        self.metabolite_feature_type = args.metabolite_feature_type
        self.protein_feature_type = args.protein_feature_type
        self.batch = Batch()

    def init_boundary_conditions(self, data, h_index, r_index):
        batch_size = len(r_index)

        # initialize queries (relation types of the given triples)
        query = self.query(r_index)

        if self.metabolite_feature_type in ["precomputed", "trained"]:
            metabolite_indx, metabolite_batch = [], []
            for i, h in enumerate(h_index):
                h = h.item() if torch.is_tensor(h) else h
                if data.metabolite_features.get(h, None) is not None:
                    metabolite_indx.append(i)
                    metabolite_batch.append(data.metabolite_features[h])

            # batch
            metabolite_batch = default_collate(metabolite_batch)

            if self.metabolite_feature_type == "precomputed":
                metabolite_features = self.metabolite_encoder(metabolite_batch)
                query[metabolite_indx] = metabolite_features["hidden"]
            elif self.metabolite_feature_type == "trained":
                # metabolite_batch = self.batch.(metabolite_batch)
                metabolite_features = self.metabolite_encoder(metabolite_batch)
                query[metabolite_indx] = metabolite_features["graph_features"]

        if self.protein_feature_type in ["precomputed", "trained"]:
            protein_indx, protein_batch = [], []
            for i, h in enumerate(h_index):
                h = h.item() if torch.is_tensor(h) else h
                if data.enzyme_features.get(h, None) is not None:
                    protein_indx.append(i)
                    protein_batch.append(data.enzyme_features[h])

            # batch
            protein_batch = default_collate(protein_batch)

            protein_features = self.protein_encoder(protein_batch)
            query[protein_indx] = protein_features["hidden"]

        index = h_index.unsqueeze(-1).expand_as(query)

        boundary = torch.zeros(
            batch_size, data.num_nodes, self.dims[0], device=h_index.device
        )
        # by the scatter operation we put query (relation) embeddings as init features of source (index) nodes
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
        return query, boundary

    def negative_sample_to_tail(self, h_index, t_index, r_index):
        # convert p(h | t, r) to p(t' | h', r')
        # h' = t, r' = r^{-1}, t' = h
        is_t_neg = (h_index == h_index[:, [0]]).all(
            dim=-1, keepdim=True
        )  # batch x 1, first half true (all true heads), second half false (all false heads)
        new_h_index = torch.where(
            is_t_neg, h_index, t_index
        )  # torch.where(condition, value if condition, value else)
        new_t_index = torch.where(is_t_neg, t_index, h_index)
        # get custom-defined inverse relations
        r_index_inverse = self.get_inverse_relation(r_index)
        new_r_index = torch.where(is_t_neg, r_index, r_index_inverse)
        return new_h_index, new_t_index, new_r_index

    def get_inverse_relation(self, r_index):
        """
        convert a tensor of relations into the metabolic inverse

        Note: Fix this reverse relation index
        Issue - most of the relations have existing reverse versions
         relation2id = {
            "is_co_reactant_of": 0, bi-directional (both directions are same relation)                  -> 0
            "is_co_product_of": 1, bi-directional (both directions are same relation)                   -> 1
            "is_co_enzyme_of": 2, bi-directional (both directions are same relation)                    -> 2
            "is_co_reactant_enzyme": 3, bi-directional (both directions are same relation)              -> 3
            "is_metabolite_reactant_for": 4, opposite direction is called is_product_of_metabolite (5)  -> 5
            "is_product_of_metabolite": 5, opposite direction is called is_metabolite_reactant_for (4)  -> 4
            "is_enzyme_reactant_for": 6, opposite direction is called is_enzyme_for_product (7)         -> 7
            "is_enzyme_for_product": 7, opposite direction is called is_enzyme_reactant_for (6)         -> 6
        }

        Args:
            r_inverse: torch.Tensor of relation indices

        Returns:
            torch.Tensor: same shape as r_index with appropriately defined inverse relations
        """
        r_index_inverse = torch.where(r_index == 4, 5, r_index)
        r_index_inverse = torch.where(r_index == 5, 4, r_index_inverse)
        r_index_inverse = torch.where(r_index == 6, 7, r_index_inverse)
        r_index_inverse = torch.where(r_index == 7, 6, r_index_inverse)
        return r_index_inverse

    @staticmethod
    def add_args(parser) -> None:
        super(Metabo_NBFNet, Metabo_NBFNet).add_args(parser)
        parser.add_argument(
            "--protein_model",
            action=set_nox_type("model"),
            type=str,
            default="identity",
            help="name of protein model",
        )
        parser.add_argument(
            "--metabolite_model",
            action=set_nox_type("model"),
            type=str,
            default="identity",
            help="name of molecule/metabolite model",
        )
