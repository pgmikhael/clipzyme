import torch
from torch import nn, einsum, broadcast_tensors
import torch.nn.functional as F
from torch.nn import SiLU

from clipzyme.utils.registry import register_object, get_object
from clipzyme.models.abstract import AbstractModel
from clipzyme.utils.classes import set_nox_type

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import copy
import os, sys
import pickle
import warnings
import hashlib
from collections import defaultdict
from rich import print as rprint
from typing import Optional, List, Union, Tuple, Any, Dict

import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, Size, OptTensor, Tensor
from torch_geometric.data import HeteroData, Data, Batch, Dataset
from torch_scatter import scatter
from torch_geometric.utils import to_dense_batch
from torch_geometric.utils.loop import add_remaining_self_loops

import math


def exists(val):
    return val is not None


class CoorsNorm(nn.Module):
    def __init__(self, eps=1e-8, scale_init=1.0):
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coors):
        norm = coors.norm(dim=-1, keepdim=True)
        normed_coors = coors / norm.clamp(min=self.eps)
        return normed_coors * self.scale


class SinusoidalEmbeddings(nn.Module):
    """A simple sinusoidal embedding layer. From Jeremy."""

    def __init__(self, dim, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        embeddings = math.log(self.theta) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        return embeddings


class EGNN_Sparse(MessagePassing):
    """ """

    propagate_type = {
        "x": Tensor,
        "edge_attr": Tensor,
        "coors": Tensor,
        "rel_coors": Tensor,
        "size": Size,
        "batch": Dict,
    }

    def __init__(self, args, **kwargs):
        self.args = args
        aggr = self.args.neighbour_aggr
        self.update_feats = self.args.update_feats
        self.update_coors = self.args.update_coors
        assert aggr in {
            "add",
            "sum",
            "max",
            "mean",
        }, "pool method must be a valid option"
        assert (
            self.update_feats or self.update_coors
        ), "you must update either features, coordinates, or both"
        kwargs.setdefault("aggr", aggr)
        super(EGNN_Sparse, self).__init__(**kwargs)
        # model params
        self.feats_dim = self.args.protein_dim
        self.feat_proj_dim = self.args.feat_proj_dim
        self.edge_attr_dim = self.args.edge_attr_dim
        self.m_dim = self.args.message_dim
        self.soft_edge = self.args.soft_edge
        self.norm_feats = self.args.norm_feats
        self.norm_coors = self.args.norm_coors
        self.norm_coors_scale_init = self.args.norm_coors_scale_init
        self.coor_weights_clamp_value = self.args.coor_weights_clamp_value

        if not self.args.use_sinusoidal:
            self.edge_input_dim = self.edge_attr_dim + 1 + (self.feats_dim * 2)
        else:
            self.edge_input_dim = self.feats_dim * 3
        self.dropout = (
            nn.Dropout(self.args.dropout) if self.args.dropout > 0 else nn.Identity()
        )

        dist_dim = (
            self.args.protein_dim
        )  # can replace if using different distance embedding
        if self.args.use_sinusoidal:
            self.dist_embedding = SinusoidalEmbeddings(dist_dim)
        else:
            self.dist_embedding = nn.Linear(1, dist_dim)  # type: ignore

        # EDGES
        # \phi_e
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.edge_input_dim, self.edge_input_dim * 2),
            self.dropout,
            SiLU(),
            nn.Linear(self.edge_input_dim * 2, self.m_dim),
            SiLU(),
        )

        self.edge_weight = (
            nn.Sequential(nn.Linear(self.m_dim, 1), nn.Sigmoid())
            if self.soft_edge
            else None
        )

        self.node_norm = (
            torch_geometric.nn.norm.LayerNorm(self.feats_dim)
            if self.norm_feats
            else None
        )
        self.coors_norm = (
            CoorsNorm(scale_init=self.norm_coors_scale_init)
            if self.norm_coors
            else nn.Identity()
        )

        # \phi_h
        self.node_mlp = (
            nn.Sequential(
                nn.Linear(self.feats_dim + self.m_dim, self.feat_proj_dim * 2),
                self.dropout,
                SiLU(),
                nn.Linear(self.feat_proj_dim * 2, self.feats_dim),
            )
            if self.update_feats
            else None
        )

        # COORS
        # \phi_x
        self.coors_mlp = (
            nn.Sequential(
                nn.Linear(self.m_dim, self.m_dim * 4),
                self.dropout,
                SiLU(),
                nn.Linear(self.m_dim * 4, 1),
            )
            if self.update_coors
            else None
        )

        self.apply(self.init_)

    def init_(self, module):
        if type(module) in {nn.Linear}:
            # seems to be needed to keep the network from exploding to NaN with greater depths
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        feats: Tensor,
        coors: Tensor,
        edge_index: Adj,
        edge_attr: OptTensor = None,
        batch: Adj = None,
        size: Size = None,
    ) -> Tensor:
        """ """
        rel_coors = coors[edge_index[0]] - coors[edge_index[1]]
        # rel_dist = (rel_coors**2).sum(dim=-1, keepdim=True)
        rel_dist = torch.sqrt((rel_coors**2).sum(dim=-1, keepdim=True))
        rel_dist = self.dist_embedding(rel_dist)
        if rel_dist.shape[1] == 1:
            rel_dist = rel_dist.squeeze(1)

        if exists(edge_attr):
            edge_attr_feats = torch.cat([edge_attr, rel_dist], dim=-1)
        else:
            edge_attr_feats = rel_dist

        hidden_out, coors_out = self.propagate(
            edge_index,
            x=feats,
            edge_attr=edge_attr_feats,
            coors=coors,
            rel_coors=rel_coors,
            size=None,
            batch=batch,
        )
        return coors_out, hidden_out

    def message(self, x_i, x_j, edge_attr) -> Tensor:
        m_ij = self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))
        return m_ij

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        """
        The initial call to start propagating messages.

        Args:
            `edge_index` holds the indices of a general (sparse)
                assignment matrix of shape :obj:`[N, M]`.
            size (tuple, optional) if none, the size will be inferred
                and assumed to be quadratic.
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """
        # try:
        #     size = self.__check_input__(edge_index, size)
        #     coll_dict = self.__collect__(self.__user_args__, edge_index, size, kwargs)
        # except:
        size = self._check_input(edge_index, size)
        coll_dict = self._collect(self._user_args, edge_index, size, kwargs)
        msg_kwargs = self.inspector.distribute("message", coll_dict)
        aggr_kwargs = self.inspector.distribute("aggregate", coll_dict)
        update_kwargs = self.inspector.distribute("update", coll_dict)

        # get messages
        m_ij = self.message(**msg_kwargs)  # graph_size x m_dim

        # update coors if specified
        if self.update_coors:
            coor_wij = self.coors_mlp(m_ij)  # graph_size x 1
            # clamp if arg is set
            if self.coor_weights_clamp_value:
                clamp_value = self.coor_weights_clamp_value
                coor_weights.clamp_(min=-clamp_value, max=clamp_value)

            # normalize if needed
            kwargs["rel_coors"] = self.coors_norm(kwargs["rel_coors"])  # graph_size x 3

            mhat_i = self.aggregate(
                coor_wij * kwargs["rel_coors"], **aggr_kwargs
            )  # graph_size / num_neighbours x 3
            coors_out = kwargs["coors"] + mhat_i  # graph_size / num_neighbours x 3
        else:
            coors_out = kwargs["coors"]

        # update feats if specified
        if self.update_feats:
            # weight the edges if arg is passed
            if self.soft_edge:
                m_ij = m_ij * self.edge_weight(m_ij)
            m_i = self.aggregate(m_ij, **aggr_kwargs)

            hidden_feats = (
                self.node_norm(kwargs["x"], kwargs["batch"])
                if self.node_norm
                else kwargs["x"]
            )
            hidden_out = self.node_mlp(
                torch.cat([hidden_feats, m_i], dim=-1)
            )  # graph_size / num_neighbours x feat_dim (esm)

            hidden_out = kwargs["x"] + hidden_out
        else:
            hidden_out = kwargs["x"]

        return self.update((hidden_out, coors_out), **update_kwargs)


@register_object("egnn_sparse_network", "model")
class EGNN_Sparse_Network(AbstractModel):
    """ """

    def __init__(self, args):
        super(EGNN_Sparse_Network, self).__init__()
        self.args = args
        self.mpnn_layers = nn.ModuleList()

        for i in range(self.args.egcl_layers):
            layer = EGNN_Sparse(args)
            self.mpnn_layers.append(layer)

    def forward(self, batch):
        """ """
        if (
            hasattr(batch, "graph")
            or "graph" in batch
            or (
                isinstance(batch, (Data, HeteroData))
                and "graph" in batch.to_dict().keys()
            )
        ):
            coors = batch["graph"]["receptor"].pos
            feats = batch["graph"]["receptor"].x
            edge_index = batch["graph"]["receptor", "contact", "receptor"].edge_index
            batch_idx = batch["graph"]["receptor"].batch
        elif (
            hasattr(batch, "receptor")
            or "receptor" in batch
            or (
                isinstance(batch, (Data, HeteroData))
                and "receptor" in batch.to_dict().keys()
            )
        ):
            coors = batch["receptor"].pos
            feats = batch["receptor"].x
            edge_index = batch["receptor", "contact", "receptor"].edge_index
            batch_idx = batch["receptor"].batch
        else:
            raise ValueError("batch must have a graph or receptor attribute")

        edge_attr = None
        bsize = batch_idx.max() + 1

        for i, egcl in enumerate(self.mpnn_layers):
            coors, feats = egcl(feats, coors, edge_index, size=bsize, batch=batch_idx)

        return feats, coors

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--pool_type",
            type=str,
            choices=["none", "sum", "mul", "mean", "min", "max"],
            default="none",
            help="Type of pooling to do to obtain graph features",
        )
        parser.add_argument(
            "--neighbour_aggr",
            type=str,
            choices=["add", "sum", "mean", "max"],
            default="sum",
            help="Type of pooling to do to obtain graph features",
        )
        parser.add_argument(
            "--edge_attr_dim",
            type=int,
            default=0,
            help="",
        )
        parser.add_argument(
            "--message_dim",
            type=int,
            default=16,
            help="",
        )
        parser.add_argument(
            "--soft_edge",
            type=int,
            default=0,
            help="",
        )
        parser.add_argument(
            "--norm_feats",
            action="store_true",
            default=False,
            help="",
        )
        parser.add_argument(
            "--norm_coors",
            action="store_true",
            default=False,
            help="",
        )
        parser.add_argument(
            "--update_feats",
            action="store_true",
            default=False,
            help="",
        )
        parser.add_argument(
            "--update_coors",
            action="store_true",
            default=False,
            help="",
        )
        parser.add_argument(
            "--norm_coors_scale_init",
            type=float,
            default=1e-2,
            help="",
        )
        parser.add_argument(
            "--coor_weights_clamp_value",
            type=float,
            default=None,
            help="",
        )
        parser.add_argument(
            "--egcl_layers",
            type=int,
            default=4,
            help="",
        )
        parser.add_argument(
            "--protein_dim",
            type=int,
            default=64,
            help="",
        )
        parser.add_argument(
            "--feat_proj_dim",
            type=int,
            default=1280,
            help="dim that features are projected to upon being fed to phi_e",
        )
        parser.add_argument(
            "--use_sinusoidal",
            action="store_true",
            default=False,
            help="whether or not to use sinusoidal embeddings for distance",
        )


@register_object("egnn_classifier", "model")
class EGNN_Classifier(AbstractModel):
    def __init__(self, args):
        super(EGNN_Classifier, self).__init__()

        self.args = args

        self.egnn = EGNN_Sparse_Network(args)

        self.esm_model = get_object(args.esm_model_name, "model")(args)
        classifier_args = copy.deepcopy(args)
        self.mlp = get_object(args.classifier_name, "model")(classifier_args)

    def forward(self, batch=None):
        output = {}
        # run ESM and update node embeddings
        batch = self.compute_embeddings(batch, self.args)
        del batch["mask_hiddens"]  # not used and is big
        feats, coors = self.egnn(batch)
        output["node_features"] = feats
        output["node_coords"] = coors
        if self.args.pool_type != "none":
            output["graph_features"] = scatter(
                feats,
                batch["graph"]["receptor"].batch,
                dim=0,
                reduce=self.args.pool_type,
            )
        else:
            output["graph_features"] = output["node_features"]

        output.update(self.mlp({"x": output["graph_features"]}))

        return output

    def compute_embeddings(self, batch, args):
        """
        Pre-compute ESM2 embeddings.
        """
        batch_hash = hashlib.md5(f"{batch}".encode()).hexdigest()
        esm_cache = os.path.join(
            args.protein_cache_path, f"{batch_hash}_cached_prot_esm.pkl"
        )
        if args.debug:
            esm_cache = esm_cache.replace(".pkl", "_debug.pkl")

        # check if we already computed embeddings
        if not args.no_graph_cache and os.path.exists(esm_cache):
            with open(esm_cache, "rb") as f:
                cropped_representations = pickle.load(f)
            self._save_esm_rep(batch, cropped_representations)
            print("Loaded cached ESM embeddings")
            return batch

        # run ESM model, use args.freeze_esm for no training
        esm_output = self.esm_model(batch)

        tokens = esm_output["tokens"]
        reps = esm_output["token_hiddens"]

        cropped_representations = []
        seq_len = [len(p) for p in batch["sequence"]]

        # TODO: use padding mask
        for i, rep in enumerate(reps):
            length = seq_len[i]
            rep_cropped = rep[
                1 : length + 1
            ]  # removes cls at idx 0 and removes eos at idx length + 2
            token_cropped = tokens[i][1 : length + 1, None]
            cropped_representations.append((rep_cropped, token_cropped))

        # dump to cache
        if not args.no_graph_cache:
            with open(esm_cache, "wb+") as f:
                pickle.dump(cropped_representations, f)

        # overwrite graph.x for each element in batch
        self._save_esm_rep(batch, cropped_representations)

        batch["mask_hiddens"] = esm_output["mask_hiddens"]

        return batch

    def _save_esm_rep(self, batch, cropped_representations):
        """
        Assign new ESM representation to graph.x INPLACE
        """
        all_graphs = batch["graph"].to_data_list()
        # cropped_representations = (rep_crop, token_crop)
        for idx, graph in enumerate(all_graphs):
            rec_graph = graph["receptor"]
            rec_graph.x = cropped_representations[idx][0]  # esm embedding
            assert len(rec_graph.pos) == len(rec_graph.x)

        batch["graph"] = Batch.from_data_list(all_graphs, None, None)

        return batch

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--classifier_name",
            type=str,
            action=set_nox_type("model"),
            default="mlp_classifier",
            help="Name of encoder to use",
        )
        parser.add_argument(
            "--esm_model_name",
            type=str,
            action=set_nox_type("model"),
            default="fair_esm2",
            help="Name of esm encoder to use",
        )
        parser.add_argument(
            "--pool_type",
            type=str,
            choices=["none", "sum", "mul", "mean", "min", "max"],
            default="sum",
            help="Type of pooling to do to obtain graph features",
        )
        parser.add_argument(
            "--neighbour_aggr",
            type=str,
            choices=["add", "sum", "mean", "max"],
            default="sum",
            help="Type of pooling to do to obtain graph features",
        )
        parser.add_argument(
            "--edge_attr_dim",
            type=int,
            default=0,
            help="",
        )
        parser.add_argument(
            "--message_dim",
            type=int,
            default=16,
            help="",
        )
        parser.add_argument(
            "--soft_edge",
            type=int,
            default=0,
            help="",
        )
        parser.add_argument(
            "--norm_feats",
            action="store_true",
            default=False,
            help="",
        )
        parser.add_argument(
            "--norm_coors",
            action="store_true",
            default=False,
            help="",
        )
        parser.add_argument(
            "--update_feats",
            action="store_true",
            default=False,
            help="",
        )
        parser.add_argument(
            "--update_coors",
            action="store_true",
            default=False,
            help="",
        )
        parser.add_argument(
            "--norm_coors_scale_init",
            type=float,
            default=1e-2,
            help="",
        )
        parser.add_argument(
            "--coor_weights_clamp_value",
            type=float,
            default=None,
            help="",
        )
        parser.add_argument(
            "--egcl_layers",
            type=int,
            default=4,
            help="",
        )


@register_object("egnn_active_site_classifier", "model")
class EGNN_Active_Site_Classifier(EGNN_Classifier):
    def __init__(self, args):
        super(EGNN_Active_Site_Classifier, self).__init__(args)

    def forward(self, batch=None):
        output = {}
        # run ESM and update node embeddings
        batch = self.compute_embeddings(batch, self.args)
        feats, coors = self.egnn(batch)
        output["node_features"] = feats
        output["node_coords"] = coors
        # reshape to batch x nodes x features
        output["graph_features"], mask = to_dense_batch(
            output["node_features"], batch=batch["graph"]["receptor"].batch
        )
        output.update(
            self.mlp({"x": output["graph_features"]})
        )  # batch x nodes x features -> batch x nodes x 1

        output["logit"] = output["logit"].squeeze(-1)
        labels = batch["residue_mask"]
        residue_mask = batch["mask_hiddens"][:, 1:-1]
        # cross entropy will ignore the padded residues (ignore_index=-100)
        labels[~residue_mask.squeeze(-1).bool()] = -100
        batch["y"] = labels

        return output

    @staticmethod
    def set_args(args) -> None:
        super(EGNN_Active_Site_Classifier, EGNN_Active_Site_Classifier).set_args(args)
        args.num_classes = 2


@register_object("egnn_active_site_conv_classifier", "model")
class EGNN_Active_Site_Conv_Classifier(EGNN_Active_Site_Classifier):
    def __init__(self, args):
        super(EGNN_Active_Site_Conv_Classifier, self).__init__(args)

    def forward(self, batch=None):
        output = {}
        # run ESM and update node embeddings
        batch = self.compute_embeddings(batch, self.args)
        feats, coors = self.egnn(batch)
        output["node_features"] = feats
        output["node_coords"] = coors
        output.update(self.mlp({"x": feats}))  # nodes x 1

        # nodes x 1
        aggr_logits = self.aggregate_neighborhood(
            output["logit"].squeeze(-1),
            batch["graph"]["receptor", "contact", "receptor"].edge_index,
        )

        # reshape to batch x nodes x 1
        logit, mask = to_dense_batch(
            aggr_logits, batch=batch["graph"]["receptor"].batch
        )

        output["logit"] = logit

        labels = batch[
            "residue_mask"
        ]  # .reshape(1, -1).squeeze(0) # flatten -> num nodes x 1
        cropped_labels = []
        for i, seq_label in enumerate(labels):
            seq_len = len(batch["sequence"][i])
            label_crop = seq_label[:seq_len]
            cropped_labels.append(label_crop)

        cropped_labels = torch.cat(cropped_labels)
        aggr_labels = self.aggregate_neighborhood(
            cropped_labels, batch["graph"]["receptor", "contact", "receptor"].edge_index
        )  # num nodes x 1
        labels, labels_mask = to_dense_batch(
            aggr_labels.unsqueeze(-1), batch=batch["graph"]["receptor"].batch
        )  # batch x nodes x 1

        residue_mask = batch["mask_hiddens"][:, 1:-1]
        # cross entropy will ignore the padded residues (ignore_index=-100)
        labels[~residue_mask.squeeze(-1).bool()] = -100
        batch["y"] = labels.squeeze(-1)
        return output

    def aggregate_neighborhood(self, logits, edge_index, reduce="max"):
        edge_idx_self_loops, edge_attr_self_loops = add_remaining_self_loops(edge_index)
        aggr_logits = scatter(
            logits[edge_idx_self_loops[0]], edge_idx_self_loops[1], reduce=reduce
        )
        return aggr_logits

    @staticmethod
    def set_args(args) -> None:
        super(
            EGNN_Active_Site_Conv_Classifier, EGNN_Active_Site_Conv_Classifier
        ).set_args(args)
        args.num_classes = 1


@register_object("egnn_mol_classifier", "model")
class EGNN_Mol_Classifier(AbstractModel):
    def __init__(self, args):
        super(EGNN_Mol_Classifier, self).__init__()

        self.args = args
        self.egnn = EGNN_Sparse_Network(args)
        self.chemprop = get_object(args.chemprop_name, "model")(args)
        classifier_args = copy.deepcopy(args)
        classifier_args.mlp_input_dim = (
            classifier_args.protein_dim + args.chemprop_hidden_dim
        )
        if self.args.use_rdkit_features:
            classifier_args.mlp_input_dim = (
                args.graph_classifier_hidden_dim + args.rdkit_features_dim
            )
        self.mlp = get_object(args.classifier_name, "model")(classifier_args)

    def forward(self, batch=None):
        output = {}
        # get protein features
        feats, coors = self.egnn(batch)
        if self.args.pool_type != "none":
            output["prot_features"] = scatter(
                feats,
                batch["receptor"].batch,
                dim=0,
                reduce=self.args.pool_type,
            )
        else:
            output["prot_features"] = feats

        # get molecule features
        if isinstance(batch["mol_data"], list):
            mol_output = self.chemprop(Batch.from_data_list(batch["mol_data"]))
        elif isinstance(batch["mol_data"], Batch):
            mol_output = self.chemprop(batch["mol_data"])
        else:
            raise ValueError(
                "mol_data must be a list of Data/HeteroData objs or Batch object"
            )
        output["mol_features"] = mol_output["graph_features"]
        if self.args.use_rdkit_features:
            features = batch["rdkit_features"].view(batch_size, -1)
            output["mol_features"] = torch.concat(
                [output["mol_features"], features], dim=-1
            )
        # concatenate and classify
        output.update(
            self.mlp(
                {
                    "x": torch.cat(
                        [output["prot_features"], output["mol_features"]], dim=-1
                    )
                }
            )
        )

        return output

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--classifier_name",
            type=str,
            action=set_nox_type("model"),
            default="mlp_classifier",
            help="Name of encoder to use",
        )
        parser.add_argument(
            "--chemprop_name",
            type=str,
            action=set_nox_type("model"),
            default="chemprop",
            help="Name of mol encoder to use",
        )
        parser.add_argument(
            "--pool_type",
            type=str,
            choices=["none", "sum", "mul", "mean", "min", "max"],
            default="sum",
            help="Type of pooling to do to obtain graph features",
        )
        parser.add_argument(
            "--neighbour_aggr",
            type=str,
            choices=["add", "sum", "mean", "max"],
            default="sum",
            help="Type of pooling to do to obtain graph features",
        )
        parser.add_argument(
            "--edge_attr_dim",
            type=int,
            default=0,
            help="",
        )
        parser.add_argument(
            "--message_dim",
            type=int,
            default=16,
            help="",
        )
        parser.add_argument(
            "--soft_edge",
            type=int,
            default=0,
            help="",
        )
        parser.add_argument(
            "--norm_feats",
            action="store_true",
            default=False,
            help="",
        )
        parser.add_argument(
            "--norm_coors",
            action="store_true",
            default=False,
            help="",
        )
        parser.add_argument(
            "--update_feats",
            action="store_true",
            default=False,
            help="",
        )
        parser.add_argument(
            "--update_coors",
            action="store_true",
            default=False,
            help="",
        )
        parser.add_argument(
            "--norm_coors_scale_init",
            type=float,
            default=1e-2,
            help="",
        )
        parser.add_argument(
            "--coor_weights_clamp_value",
            type=float,
            default=None,
            help="",
        )
        parser.add_argument(
            "--egcl_layers",
            type=int,
            default=4,
            help="",
        )
        parser.add_argument(
            "--protein_dim",
            type=int,
            default=64,
            help="",
        )
        parser.add_argument(
            "--feat_proj_dim",
            type=int,
            default=1280,
            help="dim that features are projected to upon being fed to phi_e",
        )
        parser.add_argument(
            "--use_sinusoidal",
            action="store_true",
            default=False,
            help="whether or not to use sinusoidal embeddings for distance",
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
