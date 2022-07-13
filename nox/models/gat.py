import torch
import torch.nn as nn
from nox.utils.registry import register_object, get_object
from nox.models.abstract import AbstractModel
from torch_geometric.nn.conv.gatv2_conv import GATv2Conv
from torch_scatter import scatter


@register_object("gatv2", "model")
class GATv2(AbstractModel):
    """
    https://arxiv.org/abs/2105.14491
    """

    def __init__(self, args):
        super(GATv2, self).__init__()
        self.args = args
        self.encoder = GATv2Conv(
            in_channels=args.num_chan,
            out_channels=args.gat_output_dim,
            heads=args.gat_num_heads,
            concat=args.gat_concat,
            negative_slope=args.gat_negative_slope,
            dropout=args.dropout,
            add_self_loops=args.gat_add_self_loops,
            edge_dim=args.gat_edge_dim,
            fill_value=args.gat_fill_value,
            bias=args.gat_bias,
            share_weights=args.gat_share_weights,
        )

    def forward(self, data):
        output = {}
        node_features = data.x
        edge_index = data.edge_index
        edge_features = data.edge_attr
        num_nodes = len(node_features)
        # default: (num_nodes, num_heads * out_chan );
        # bipartite (num_target_nodes, num_heads * out_chan )
        encoded_features = self.encoder.forward(
            node_features, edge_index, edge_features
        )
        encoded_features = encoded_features.view(num_nodes, self.num_heads, -1).mean(1)
        graph_features = scatter(encoded_features, data.batch, dim=0, reduce="add")

        output["node_features"] = encoded_features
        output["graph_features"] = graph_features
        return output

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--gat_output_dim", type=int, nargs="*", help="Size of each output sample."
        )
        parser.add_argument(
            "--gat_num_heads",
            type=int,
            default=1,
            help="Number of multi-head-attentions.",
        )
        parser.add_argument(
            "--gat_concat",
            action="store_true",
            default=False,
            help="If set to False, the multi-head attentions are averaged instead of concatenated.",
        )
        parser.add_argument(
            "--gat_negative_slope",
            type=float,
            default=0.2,
            help="LeakyReLU angle of the negative slope.",
        )
        parser.add_argument(
            "--gat_add_self_loops",
            action="store_true",
            default=False,
            help="If set to False, will not add self-loops to the input graph.",
        )
        parser.add_argument(
            "--gat_edge_dim",
            type=int,
            default=None,
            help="Edge feature dimensionality (in case there are any).",
        )
        parser.add_argument(
            "--gat_fill_value",
            type=str,
            default="mean",
            help="The way to generate edge features of self-loops (in case edge_dim != None). If given as float or torch.Tensor, edge features of self-loops will be directly given by fill_value. If given as str, edge features of self-loops are computed by aggregating all features of edges that point to the specific node, according to a reduce operation. [add, mean, min, max, mul].",
        )
        parser.add_argument(
            "--gat_bias",
            action="store_true",
            default=False,
            help="If set to False, the layer will not learn an additive bias.",
        )
        parser.add_argument(
            "--gat_share_weights",
            action="store_true",
            default=False,
            help="If set to True, the same matrix will be applied to the source and the target node of every edge",
        )
