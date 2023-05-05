"""Combines models/layers.py and models/transformer_model.py"""
import math

import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn import functional as F
from torch import Tensor

from nox.utils.digress import diffusion_utils
from nox.utils.registry import register_object
from nox.models.abstract import AbstractModel
import json


class Xtoy(nn.Module):
    def __init__(self, dx, dy):
        """Map node features to global features"""
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy)

    def forward(self, X):
        """X: bs, n, dx."""
        m = X.mean(dim=1)
        mi = X.min(dim=1)[0]
        ma = X.max(dim=1)[0]
        std = X.std(dim=1)  # ! cannot have one node in graph
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out


class Etoy(nn.Module):
    def __init__(self, d, dy):
        """Map edge features to global features."""
        super().__init__()
        self.lin = nn.Linear(4 * d, dy)

    def forward(self, E):
        """E: bs, n, n, de
        Features relative to the diagonal of E could potentially be added.
        """
        m = E.mean(dim=(1, 2))
        mi = E.min(dim=2)[0].min(dim=1)[0]
        ma = E.max(dim=2)[0].max(dim=1)[0]
        std = torch.std(E, dim=(1, 2))  # ! cannot have one node in graph
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out


class XEyTransformerLayer(nn.Module):
    """Transformer that updates node, edge and global features
    d_x: node features
    d_e: edge features
    dz : global features
    n_head: the number of heads in the multi_head_attention
    dim_feedforward: the dimension of the feedforward network model after self-attention
    dropout: dropout probablility. 0 to disable
    layer_norm_eps: eps value in layer normalizations.
    """

    def __init__(
        self,
        dx: int,
        de: int,
        dy: int,
        n_head: int,
        dim_ffX: int = 2048,
        dim_ffE: int = 128,
        dim_ffy: int = 2048,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        device=None,
        dtype=None,
    ) -> None:
        kw = {"device": device, "dtype": dtype}
        super().__init__()

        self.self_attn = NodeEdgeBlock(dx, de, dy, n_head, **kw)

        self.linX1 = Linear(dx, dim_ffX, **kw)
        self.linX2 = Linear(dim_ffX, dx, **kw)
        self.normX1 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.normX2 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.dropoutX1 = Dropout(dropout)
        self.dropoutX2 = Dropout(dropout)
        self.dropoutX3 = Dropout(dropout)

        self.linE1 = Linear(de, dim_ffE, **kw)
        self.linE2 = Linear(dim_ffE, de, **kw)
        self.normE1 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.normE2 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.dropoutE1 = Dropout(dropout)
        self.dropoutE2 = Dropout(dropout)
        self.dropoutE3 = Dropout(dropout)

        self.lin_y1 = Linear(dy, dim_ffy, **kw)
        self.lin_y2 = Linear(dim_ffy, dy, **kw)
        self.norm_y1 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.norm_y2 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.dropout_y1 = Dropout(dropout)
        self.dropout_y2 = Dropout(dropout)
        self.dropout_y3 = Dropout(dropout)

        self.activation = F.relu

    def forward(self, X: Tensor, E: Tensor, y, node_mask: Tensor):
        """Pass the input through the encoder layer.
        X: (bs, n, d)
        E: (bs, n, n, d)
        y: (bs, dy)
        node_mask: (bs, n) Mask for the src keys per batch (optional)
        Output: newX, newE, new_y with the same shape.
        """

        newX, newE, new_y = self.self_attn(X, E, y, node_mask=node_mask)

        newX_d = self.dropoutX1(newX)
        X = self.normX1(X + newX_d)

        newE_d = self.dropoutE1(newE)
        E = self.normE1(E + newE_d)

        new_y_d = self.dropout_y1(new_y)
        y = self.norm_y1(y + new_y_d)

        ff_outputX = self.linX2(self.dropoutX2(self.activation(self.linX1(X))))
        ff_outputX = self.dropoutX3(ff_outputX)
        X = self.normX2(X + ff_outputX)

        ff_outputE = self.linE2(self.dropoutE2(self.activation(self.linE1(E))))
        ff_outputE = self.dropoutE3(ff_outputE)
        E = self.normE2(E + ff_outputE)

        ff_output_y = self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))
        ff_output_y = self.dropout_y3(ff_output_y)
        y = self.norm_y2(y + ff_output_y)

        return X, E, y


class NodeEdgeBlock(nn.Module):
    """Self attention layer that also updates the representations on the edges."""

    def __init__(self, dx, de, dy, n_head, **kwargs):
        super().__init__()
        assert dx % n_head == 0, f"dx: {dx} -- nhead: {n_head}"
        self.dx = dx
        self.de = de
        self.dy = dy
        self.df = int(dx / n_head)
        self.n_head = n_head

        # Attention
        self.q = Linear(dx, dx)
        self.k = Linear(dx, dx)
        self.v = Linear(dx, dx)

        # FiLM E to X
        self.e_add = Linear(de, dx)
        self.e_mul = Linear(de, dx)

        # FiLM y to E
        self.y_e_mul = Linear(dy, dx)  # Warning: here it's dx and not de
        self.y_e_add = Linear(dy, dx)

        # FiLM y to X
        self.y_x_mul = Linear(dy, dx)
        self.y_x_add = Linear(dy, dx)

        # Process y
        self.y_y = Linear(dy, dy)
        self.x_y = Xtoy(dx, dy)
        self.e_y = Etoy(de, dy)

        # Output layers
        self.x_out = Linear(dx, dx)
        self.e_out = Linear(dx, de)
        self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))

    def forward(self, X, E, y, node_mask):
        """
        :param X: bs, n, d        node features
        :param E: bs, n, n, d     edge features
        :param y: bs, dz           global features
        :param node_mask: bs, n
        :return: newX, newE, new_y with the same shape.
        """
        x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1

        # 1. Map X to keys and queries
        Q = self.q(X) * x_mask  # (bs, n, dx)
        K = self.k(X) * x_mask  # (bs, n, dx)
        diffusion_utils.assert_correctly_masked(Q, x_mask)
        # 2. Reshape to (bs, n, n_head, df) with dx = n_head * df

        Q = Q.reshape((Q.size(0), Q.size(1), self.n_head, self.df))
        K = K.reshape((K.size(0), K.size(1), self.n_head, self.df))

        Q = Q.unsqueeze(2)  # (bs, 1, n, n_head, df)
        K = K.unsqueeze(1)  # (bs, n, 1, n head, df)

        # Compute unnormalized attentions. Y is (bs, n, n, n_head, df)
        Y = Q * K
        Y = Y / math.sqrt(Y.size(-1))
        diffusion_utils.assert_correctly_masked(Y, (e_mask1 * e_mask2).unsqueeze(-1))

        E1 = self.e_mul(E) * e_mask1 * e_mask2  # bs, n, n, dx
        E1 = E1.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        E2 = self.e_add(E) * e_mask1 * e_mask2  # bs, n, n, dx
        E2 = E2.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        # Incorporate edge features to the self attention scores.
        Y = Y * (E1 + 1) + E2  # (bs, n, n, n_head, df)

        # Incorporate y to E
        newE = Y.flatten(start_dim=3)  # bs, n, n, dx
        ye1 = self.y_e_add(y).unsqueeze(1).unsqueeze(1)  # bs, 1, 1, de
        ye2 = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
        newE = ye1 + (ye2 + 1) * newE

        # Output E
        newE = self.e_out(newE) * e_mask1 * e_mask2  # bs, n, n, de
        diffusion_utils.assert_correctly_masked(newE, e_mask1 * e_mask2)

        # Compute attentions. attn is still (bs, n, n, n_head, df)
        attn = F.softmax(Y, dim=2)

        V = self.v(X) * x_mask  # bs, n, dx
        V = V.reshape((V.size(0), V.size(1), self.n_head, self.df))
        V = V.unsqueeze(1)  # (bs, 1, n, n_head, df)

        # Compute weighted values
        weighted_V = attn * V
        weighted_V = weighted_V.sum(dim=2)

        # Send output to input dim
        weighted_V = weighted_V.flatten(start_dim=2)  # bs, n, dx

        # Incorporate y to X
        yx1 = self.y_x_add(y).unsqueeze(1)
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newX = yx1 + (yx2 + 1) * weighted_V

        # Output X
        newX = self.x_out(newX) * x_mask
        diffusion_utils.assert_correctly_masked(newX, x_mask)

        # Process y based on X axnd E
        y = self.y_y(y)
        e_y = self.e_y(E)
        x_y = self.x_y(X)
        new_y = y + x_y + e_y
        new_y = self.y_out(new_y)  # bs, dy

        return newX, newE, new_y


@register_object("graph_transformer", "model")
class GraphTransformer(AbstractModel):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """

    @staticmethod
    def add_args(parser) -> None:
        parser.add_argument(
            "--gt_n_layers", type=int, default=2, help="Number of layers"
        )

    def __init__(self, args):
        super().__init__()

        input_dims = args.dataset_statistics.input_dims
        hidden_mlp_dims = args.gt_hidden_mlp_dims
        hidden_dims = args.gt_hidden_dims
        output_dims = args.dataset_statistics.output_dims

        self.n_layers = args.gt_n_layers
        self.out_dim_X = output_dims["X"]
        self.out_dim_E = output_dims["E"]
        self.out_dim_y = output_dims["y"]

        self.mlp_in_X = nn.Sequential(
            nn.Linear(input_dims["X"], hidden_mlp_dims["X"]),
            nn.ReLU(),
            nn.Linear(hidden_mlp_dims["X"], hidden_dims["dx"]),
            nn.ReLU(),
        )

        self.mlp_in_E = nn.Sequential(
            nn.Linear(input_dims["E"], hidden_mlp_dims["E"]),
            nn.ReLU(),
            nn.Linear(hidden_mlp_dims["E"], hidden_dims["de"]),
            nn.ReLU(),
        )

        self.mlp_in_y = nn.Sequential(
            nn.Linear(input_dims["y"], hidden_mlp_dims["y"]),
            nn.ReLU(),
            nn.Linear(hidden_mlp_dims["y"], hidden_dims["dy"]),
            nn.ReLU(),
        )

        self.tf_layers = nn.ModuleList(
            [
                XEyTransformerLayer(
                    dx=hidden_dims["dx"],
                    de=hidden_dims["de"],
                    dy=hidden_dims["dy"],
                    n_head=hidden_dims["n_head"],
                    dim_ffX=hidden_dims["dim_ffX"],
                    dim_ffE=hidden_dims["dim_ffE"],
                )
                for i in range(self.n_layers)
            ]
        )

        self.mlp_out_X = nn.Sequential(
            nn.Linear(hidden_dims["dx"], hidden_mlp_dims["X"]),
            nn.ReLU(),
            nn.Linear(hidden_mlp_dims["X"], output_dims["X"]),
        )

        self.mlp_out_E = nn.Sequential(
            nn.Linear(hidden_dims["de"], hidden_mlp_dims["E"]),
            nn.ReLU(),
            nn.Linear(hidden_mlp_dims["E"], output_dims["E"]),
        )

        if output_dims["y"] > 0:
            self.mlp_out_y = nn.Sequential(
                nn.Linear(hidden_dims["dy"], hidden_mlp_dims["y"]),
                nn.ReLU(),
                nn.Linear(hidden_mlp_dims["y"], output_dims["y"]),
            )

    def forward(self, batch):
        X, E, y, node_mask = (
            batch["X"],
            batch["E"],
            batch["y"],
            batch["node_mask"],
        )

        bs, n = X.shape[0], X.shape[1]

        diag_mask = torch.eye(n)
        diag_mask = ~diag_mask.type_as(E).bool()
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        X_to_out = X[..., : self.out_dim_X]
        E_to_out = E[..., : self.out_dim_E]
        y_to_out = y[..., : self.out_dim_y]

        new_E = self.mlp_in_E(E)
        new_E = (new_E + new_E.transpose(1, 2)) / 2
        after_in = diffusion_utils.PlaceHolder(
            X=self.mlp_in_X(X), E=new_E, y=self.mlp_in_y(y)
        ).mask(node_mask)
        X, E, y = after_in.X, after_in.E, after_in.y

        for layer in self.tf_layers:
            X, E, y = layer(X, E, y, node_mask)

        X = self.mlp_out_X(X)
        E = self.mlp_out_E(E)

        X = X + X_to_out
        E = (E + E_to_out) * diag_mask

        if hasattr(self, "mlp_out_y"):
            y = self.mlp_out_y(y)
            y = y + y_to_out
        else:
            y = y_to_out

        E = 1 / 2 * (E + torch.transpose(E, 1, 2))

        return diffusion_utils.PlaceHolder(X=X, E=E, y=y).mask(node_mask)

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        super(GraphTransformer, GraphTransformer).add_args(parser)
        parser.add_argument(
            "--gt_hidden_mlp_dims",
            type=json.loads,
            default='{ "X": 17, "E": 18, "y": 19 }',
            help="dimension of ffns",
        )
        parser.add_argument(
            "--gt_hidden_dims",
            type=json.loads,
            default='{"dx": 20, "de": 21, "dy": 22, "n_head": 5, "dim_ffX": 23, "dim_ffE": 24, "dim_ffy": 25}',
            help="dimension of transformer hiddens",
        )
        parser.add_argument(
            "--gt_n_layers",
            type=int,
            default=2,
            help="num layers",
        )


from nox.models.chemprop import DMPNNEncoder
from torch_geometric.utils import (
    dense_to_sparse,
    to_dense_adj,
    remove_self_loops,
    to_dense_batch,
)
from typing import NamedTuple


class GraphData(NamedTuple):
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    num_nodes: torch.Tensor
    batch: torch.Tensor


def multidim_dense_to_sparse(adj, node_batch_ids):
    """
    adj: adjacency matrix [batch_size, num_nodes, num_nodes, edge_dim]
    """
    # Find non-zero edges in `adj` with multi-dimensional edge_features

    B, N, _, D = adj.shape
    mask = torch.eye(N).unsqueeze(0).unsqueeze(-1).repeat(B, 1, 1, D).to(adj.device)
    adj = adj * (1 - mask)  # remove self loop

    adj2 = adj.abs().sum(dim=-1)
    index = adj2.nonzero(as_tuple=True)
    edge_attr = adj[index]
    batch = index[0] * adj.size(-2)
    # index = (batch + index[1], batch + index[2])

    # make edge_index such that indices correspond to num nodes in each graph
    graph_num_edges = torch.bincount(index[0])  # number of edges in each graph
    graph_sizes = torch.bincount(node_batch_ids)  # number of nodes in each graph
    graph_sizes_cum = torch.cumsum(
        graph_sizes, 0
    )  # cumulative sum, used to shift indices by constant factor
    graph_sizes_shifted = torch.zeros_like(graph_sizes_cum)
    graph_sizes_shifted[1:] = graph_sizes_cum[:-1]
    graph_sizes_shifted = graph_sizes_shifted.repeat_interleave(graph_num_edges)
    index = (graph_sizes_shifted + index[1], graph_sizes_shifted + index[2])

    edge_index = torch.stack(index, dim=0)
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    return edge_index, edge_attr


def dense_to_sparse_nodes(node_features, node_mask):
    """
    node_features: [batch_size, max_num_nodes, node_dim]
    node_mask: [batch_size, max_num_nodes]: 1 if true node
    """
    batch_size = node_features.shape[0]
    x = node_features[node_mask]
    arr = torch.arange(batch_size).to(node_features.device)
    batch = torch.repeat_interleave(arr, node_mask.sum(-1))
    return x, batch


@register_object("chemprop_denoiser", "model")
class ChempropDenoiser(DMPNNEncoder):
    def __init__(self, args):
        input_dims = args.dataset_statistics.input_dims
        args.chemprop_node_dim = input_dims["X"] + input_dims["y"]
        args.chemprop_edge_dim = input_dims["E"]  # + input_dims["y"]

        super(ChempropDenoiser, self).__init__(args)

        output_dims = args.dataset_statistics.output_dims
        self.mlp_out_X = nn.Sequential(
            nn.Linear(args.chemprop_hidden_dim, args.chemprop_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.chemprop_hidden_dim, output_dims["X"]),
        )

        self.mlp_out_E = nn.Sequential(
            nn.Linear(args.chemprop_hidden_dim, args.chemprop_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.chemprop_hidden_dim, output_dims["E"]),
        )

        self.out_dim_X = output_dims["X"]
        self.out_dim_E = output_dims["E"]
        self.out_dim_y = output_dims["y"]

    def forward(self, batch):
        X, E, y, node_mask = (
            batch["X"],
            batch["E"],
            batch["y"],
            batch["node_mask"],
        )

        B, N, d = X.shape

        diag_mask = torch.eye(N)
        diag_mask = ~diag_mask.type_as(E).bool()
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, -1)

        X_to_out = X[..., : self.out_dim_X]
        E_to_out = E[..., : self.out_dim_E]
        y_to_out = y[..., : self.out_dim_y]

        # add y
        y_for_x = y.unsqueeze(1).repeat(1, N, 1)
        X = torch.concat((X, y_for_x), -1)

        # process into correct shapes: dense to sparse
        # x: [num_nodes, num_features]
        # edge_index: [2, num_edges]
        # edge_attr: [edge_attr, num_features]
        sparse_x, node_batch_ids = dense_to_sparse_nodes(X, node_mask)
        edge_index, edge_attr = multidim_dense_to_sparse(E, node_batch_ids)

        # y_for_e = y[:,None,None].repeat(1, N, N, 1)
        # E = torch.concat((E,y_for_e), -1)

        # prepare input data
        data = GraphData(
            x=sparse_x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=node_mask.sum().item(),
            batch=node_batch_ids,
        )

        # pass through chemprop
        output = super().forward(data)
        node_features = output["node_features"]
        edge_features = output["edge_features"]

        # process into correct shapes: sparse to dense
        max_num_nodes = X.shape[-2]
        X, _ = to_dense_batch(x=node_features, batch=node_batch_ids)
        E = to_dense_adj(
            edge_index=edge_index,
            batch=node_batch_ids,
            edge_attr=edge_features,
            max_num_nodes=max_num_nodes,
        )

        # classify
        X = self.mlp_out_X(X)
        E = self.mlp_out_E(E)

        X = X + X_to_out
        E = (E + E_to_out) * diag_mask
        E = 1 / 2 * (E + torch.transpose(E, 1, 2))
        y = y_to_out

        return diffusion_utils.PlaceHolder(X=X, E=E, y=y).mask(node_mask)
