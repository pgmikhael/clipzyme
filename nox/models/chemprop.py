"""
Adapted from: https://github.com/itakigawa/pyg_chemprop/blob/main/pyg_chemprop.py
"""

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter_sum
from tqdm import tqdm
from nox.utils.registry import register_object
from nox.models.abstract import AbstractModel


def get_reverse_edge_indices(edge_index):
    revedge_index = torch.zeros(edge_index.shape[1]).long()
    for k, (i, j) in enumerate(zip(*edge_index)):
        edge_to_i = edge_index[1] == i
        edge_from_j = edge_index[0] == j
        revedge_index[k] = torch.where(edge_to_i & edge_from_j)[0]
    return revedge_index


def directed_mp(message, edge_index, revedge_index):
    m = scatter_sum(message, edge_index[1], dim=0)
    m_all = m[edge_index[0]]
    m_rev = message[revedge_index]
    return m_all - m_rev


def aggregate_at_nodes(num_nodes, message, edge_index):
    m = scatter_sum(message, edge_index[1], dim=0, dim_size=num_nodes)
    return m[torch.arange(num_nodes)]


@register_object("chemprop", "model")
class DMPNNEncoder(AbstractModel):
    def __init__(self, args):
        super(DMPNNEncoder, self).__init__()
        hidden_size = args.chemprop_hidden_dim
        node_fdim = args.chemprop_node_dim
        edge_fdim = args.chemprop_edge_dim

        self.act_func = nn.ReLU()
        self.W1 = nn.Linear(node_fdim + edge_fdim, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W3 = nn.Linear(node_fdim + hidden_size, hidden_size, bias=True)
        self.depth = args.chemprop_num_layers

        self.pool_type = args.chemprop_pool

    def forward(self, data):
        x, edge_index, edge_attr, num_nodes, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.num_nodes,
            data.batch,
        )

        # get indices of reverse direction
        revedge_index = get_reverse_edge_indices(edge_index)

        # initialize messages on edges
        init_msg = torch.cat([x[edge_index[0]], edge_attr], dim=1).float()
        h0 = self.act_func(self.W1(init_msg))

        # directed message passing over edges
        h = h0
        for _ in range(self.depth - 1):
            m = directed_mp(h, edge_index, revedge_index)
            h = self.act_func(h0 + self.W2(m))

        # aggregate in-edge messages at nodes
        v_msg = aggregate_at_nodes(num_nodes, h, edge_index)

        z = torch.cat([x, v_msg], dim=1)
        node_attr = self.act_func(self.W3(z))

        # readout: pyg global pooling
        output = {}
        output["node_features"] = node_attr
        if self.pool_type != "none":
            output["graph_features"] = global_mean_pool(node_attr, batch)
            output["hidden"] = output["graph_features"]

        return output

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--chemprop_num_layers",
            type=int,
            default=1,
            help="Number of layers in GNN, equivalently number of convolution iterations.",
        )
        parser.add_argument(
            "--chemprop_hidden_dim",
            type=int,
            default=None,
            help="Dimension of hidden layers (node features)",
        )
        parser.add_argument(
            "--chemprop_node_dim",
            type=int,
            default=None,
            help="Node feature dimensionality.",
        )
        parser.add_argument(
            "--chemprop_edge_dim",
            type=int,
            default=None,
            help="Edge feature dimensionality (in case there are any).",
        )
        parser.add_argument(
            "--chemprop_pool",
            type=str,
            choices=["none", "sum", "mul", "mean", "min", "max"],
            default="sum",
            help="Type of pooling to do to obtain graph features",
        )
