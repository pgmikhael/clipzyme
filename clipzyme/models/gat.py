import torch.nn as nn
import torch.nn.functional as F
from clipzyme.utils.registry import register_object
from clipzyme.models.abstract import AbstractModel
from torch_geometric.nn.conv.gatv2_conv import GATv2Conv
from torch_scatter import scatter
from clipzyme.utils.pyg import unbatch


@register_object("gatv2conv", "model")
class GATv2Op(AbstractModel):
    """
    https://arxiv.org/abs/2105.14491
    """

    def __init__(self, args):
        super(GATv2Op, self).__init__()
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
        self.use_edge_features = args.gat_edge_dim is not None

    def forward(self, graph):
        output = {}
        node_features = graph.x
        edge_index = graph.edge_index
        edge_features = graph.edge_attr
        num_nodes = len(node_features)
        # default: (num_nodes, num_heads * out_chan );
        # bipartite (num_target_nodes, num_heads * out_chan )
        if self.use_edge_features:
            encoded_features = self.encoder.forward(
                node_features.float(), edge_index, edge_features.float()
            )
        else:
            encoded_features = self.encoder.forward(node_features.float(), edge_index)
        graph_features = scatter(encoded_features, graph.batch, dim=0, reduce="add")
        encoded_features = unbatch(encoded_features, graph.batch)
        output["node_features"] = encoded_features
        output["graph_features"] = graph_features
        output["hidden"] = output["graph_features"]
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


@register_object("gatv2", "model")
class GAT(AbstractModel):
    """
    https://arxiv.org/abs/2105.14491
    """

    def __init__(self, args):
        super().__init__()

        n_hidden = args.gat_hidden_dim
        self.num_layers = args.gat_num_layers
        n_heads = args.gat_num_heads
        self.use_edge_features = args.gat_edge_dim is not None

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.node_encoder = nn.Linear(args.gat_node_dim, n_hidden)

        for i in range(self.num_layers):
            in_hidden = n_hidden
            out_hidden = n_hidden
            # bias = i == n_layers - 1
            if (i == 0) and (self.use_edge_features):
                layer = GATv2Conv(
                    in_channels=in_hidden,
                    out_channels=out_hidden // n_heads,
                    heads=n_heads,
                    concat=True,  # args.gat_concat,
                    negative_slope=args.gat_negative_slope,
                    dropout=args.dropout,
                    add_self_loops=args.gat_add_self_loops,
                    edge_dim=args.gat_edge_dim,
                    fill_value=args.gat_fill_value,
                    bias=args.gat_bias,
                    share_weights=args.gat_share_weights,
                )
            else:
                layer = GATv2Conv(
                    in_channels=in_hidden,
                    out_channels=out_hidden // n_heads,
                    heads=n_heads,
                    concat=True,  # args.gat_concat,
                    negative_slope=args.gat_negative_slope,
                    dropout=args.dropout,
                    add_self_loops=args.gat_add_self_loops,
                    fill_value=args.gat_fill_value,
                    bias=args.gat_bias,
                    share_weights=args.gat_share_weights,
                )
            self.convs.append(layer)
            self.norms.append(nn.BatchNorm1d(out_hidden))

        self.input_drop = nn.Dropout(args.dropout)
        self.dropout = nn.Dropout(args.dropout)

        self.pool_type = args.gat_pool

    def forward(self, graph):
        output = {}

        h = graph.x.float()
        h = self.node_encoder(h)
        h = F.relu(h, inplace=True)
        h = self.input_drop(h)

        h_last = None

        for i in range(self.num_layers):
            if (i == 0) and self.use_edge_features:
                h = self.convs[i](h, graph.edge_index, graph.edge_attr.float())
            else:
                h = self.convs[i](h, graph.edge_index)

            if h_last is not None:
                h += h_last[: h.shape[0], :]

            h_last = h

            h = self.norms[i](h)
            h = F.relu(h, inplace=True)
            h = self.dropout(h)

        output["node_features"] = h
        if self.pool_type != "none":
            output["graph_features"] = scatter(
                h, graph.batch, dim=0, reduce=self.pool_type
            )
            output["hidden"] = output["graph_features"]
        return output

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--gat_num_layers",
            type=int,
            default=1,
            help="Number of layers in GNN, equivalently number of convolution iterations.",
        )
        parser.add_argument(
            "--gat_hidden_dim",
            type=int,
            default=None,
            help="Dimension of hidden layers (node features)",
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
            "--gat_node_dim",
            type=int,
            default=None,
            help="Node feature dimensionality.",
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
        parser.add_argument(
            "--gat_pool",
            type=str,
            choices=["none", "sum", "mul", "mean", "min", "max"],
            default="sum",
            help="Type of pooling to do to obtain graph features",
        )
