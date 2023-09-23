import torch
import torch.nn as nn
import torch.nn.functional as F
from nox.utils.registry import get_object, register_object
from nox.utils.classes import set_nox_type
from nox.utils.pyg import unbatch
from nox.utils.wln_processing import (
    generate_candidates_from_scores,
    get_batch_candidate_bonds,
    robust_edit_mol,
)
from nox.models.abstract import AbstractModel
from torch_scatter import scatter, scatter_add
from torch_geometric.data import HeteroData, Data, Batch
from torch_geometric.utils import to_dense_batch, to_dense_adj, dense_to_sparse
from collections import defaultdict
from nox.models.gat import GAT
from nox.models.chemprop import WLNEncoder, DMPNNEncoder
from rdkit import Chem
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
        self.P_a = nn.Linear(args.gat_hidden_dim, args.gat_hidden_dim)
        self.P_b = nn.Linear(args.gat_complete_edge_dim, args.gat_hidden_dim)
        self.U = nn.Linear(args.gat_hidden_dim, 1)

    def forward(self, node_feats, graph):
        """Compute node contexts with global attention

        node_feats: N x F, where N is number of nodes, F is feature dimension
        graph: batched graph with edges of complete graph
        """
        # Batch index: N, mapping each node to corresponding graph index
        edge_index_complete = graph.edge_index_complete
        edge_attr_complete = graph.edge_attr_complete
        batch_index = graph.batch

        # Project features
        node_feats_transformed = self.P_a(node_feats)  # N x F
        edge_feats_complete = self.P_b(edge_attr_complete.float())  # E x F

        # convert to dense adj: E x F -> N x N x F
        dense_edge_attr = to_dense_adj(
            edge_index=edge_index_complete, edge_attr=edge_feats_complete
        ).squeeze(0)

        # node_features: sparse batch: N x D
        pairwise_node_feats = (
            node_feats_transformed.unsqueeze(1) + node_feats_transformed
        )  # N x N x F

        # Compute attention scores
        scores = torch.sigmoid(
            self.U(F.relu(pairwise_node_feats + dense_edge_attr))
        ).squeeze(-1)

        # Mask attention scores to prevent attention to nodes in different graphs
        mask = batch_index[:, None] != batch_index[None, :]  # N x N
        weights = scores.masked_fill(mask, 0)

        # Apply attention weights
        weighted_feats = torch.matmul(weights, node_feats)  # N x F

        return weighted_feats  # node_contexts


@register_object("gatv2_globalattn", "model")
class GATWithGlobalAttn(GAT):
    def __init__(self, args):
        super().__init__(args)
        self.global_attention = get_object(args.attn_type, "model")(args)

    def forward(self, graph):
        output = super().forward(graph)  # Graph NN (GAT) Local Network

        weighted_node_feats = self.global_attention(
            output["node_features"], graph
        )  # EQN 6

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
            help="type of global attention to use",
        )


@register_object("reaction_center_net", "model")
class ReactivityCenterNet(AbstractModel):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.gat_global_attention = get_object(args.gat_type, "model")(
            args
        )  # GATWithGlobalAttn(args)
        self.M_a = nn.Linear(args.gat_hidden_dim, args.gat_hidden_dim)
        self.M_b = nn.Linear(args.gat_complete_edge_dim, args.gat_hidden_dim)
        self.lin = nn.Linear(2 * args.gat_hidden_dim, args.gat_hidden_dim)
        self.U = nn.Sequential(
            nn.ReLU(), nn.Linear(args.gat_hidden_dim, args.num_predicted_bond_types)
        )

    def forward(self, batch):
        gat_output = self.gat_global_attention(
            batch["reactants"]
        )  # GAT + Global Attention over node features
        cs = gat_output["node_features"]
        c_tildes = gat_output["node_features_attn"]  # node contexts
        c_final = self.lin(
            torch.cat([cs, c_tildes], dim=-1)
        )  # N x 2*hidden_dim -> N x hidden_dim

        s_uv = self.forward_helper(
            c_final,
            batch["reactants"]["edge_index_complete"],
            batch["reactants"]["edge_attr_complete"],
            batch["reactants"]["batch"],
        )

        # precompute for top k metric
        candidate_bond_changes = get_batch_candidate_bonds(
            batch["reaction"], s_uv.detach(), batch["reactants"].batch
        )
        # make bonds that are "4" -> "1.5"
        for i in range(len(candidate_bond_changes)):
            candidate_bond_changes[i] = [
                (elem[0], elem[1], 1.5, elem[3]) if elem[2] == 4 else elem
                for elem in candidate_bond_changes[i]
            ]

        batch_real_bond_changes = []
        for i in range(len(batch["reactants"]["bond_changes"])):
            reaction_real_bond_changes = []
            for elem in batch["reactants"]["bond_changes"][i]:
                reaction_real_bond_changes.append(tuple(elem))
            batch_real_bond_changes.append(reaction_real_bond_changes)

        assert len(candidate_bond_changes) == len(batch_real_bond_changes)

        return {
            "s_uv": s_uv,
            "candidate_bond_changes": candidate_bond_changes,
            "real_bond_changes": batch_real_bond_changes,
            "c_final": c_final,
        }

    def forward_helper(self, node_features, edge_indices, edge_attr, batch_indices):
        # GAT with global attention
        node_features = self.M_a(node_features)  # N x hidden_dim -> N x hidden_dim
        edge_attr = self.M_b(edge_attr.float())  # E x 5 -> E x hidden_dim

        # convert to dense adj: E x hidden_dim -> N x N x hidden_dim
        dense_edge_attr = to_dense_adj(
            edge_index=edge_indices, edge_attr=edge_attr
        ).squeeze(0)

        # node_features: sparse batch: N x D
        pairwise_node_feats = node_features.unsqueeze(1) + node_features  # N x N x D
        # edge_attr: bond features: N x N x D
        s = self.U(dense_edge_attr + pairwise_node_feats).squeeze(-1)  # N x N
        # removed this line since the sizes become inconsistent later
        # s, mask = to_dense_batch(s, batch_indices) # B x max_batch_N x N x num_predicted_bond_types

        # make symmetric
        s = (s + s.transpose(0, 1)) / 2
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
            help="number of bond types to predict, this is t in the paper",
        )
        parser.add_argument(
            "--gat_type",
            type=str,
            action=set_nox_type("model"),
            default="gatv2_globalattn",
            help="Type of gat to use, mainly to init args",
        )
        parser.add_argument(
            "--topk_bonds",
            nargs="+",
            type=int,
            default=[1, 3, 5],
            help="topk bonds to consider for accuracy metric",
        )
        parser.add_argument(
            "--gat_complete_edge_dim",
            type=int,
            default=5,
            help="dimension of edges in complete graph",
        )


@register_object("wldn", "model")
class WLDN(AbstractModel):
    def __init__(self, args):
        super().__init__()
        self.args = args
        try:
            state_dict = torch.load(args.reactivity_model_path)
            self.reactivity_net = get_object(args.reactivity_net_type, "model")(
                state_dict["hyper_parameters"]["args"]
            )
            self.reactivity_net.load_state_dict(
                {
                    k[len("model.") :]: v
                    for k, v in state_dict["state_dict"].items()
                    if k.startswith("model")
                }
            )
            self.reactivity_net.requires_grad_(False)
        except:
            self.reactivity_net = get_object(args.reactivity_net_type, "model")(
                args
            ).requires_grad_(False)
            print("Could not load pretrained model")
        self.reactivity_net.eval()

        self.add_scores_features = getattr(args, "add_scores_features", True)

        wln_diff_args = copy.deepcopy(args)

        if args.use_wln_encoder:
            # WLNEncoder
            self.wln = WLNEncoder(args)  # WLN for mol representation
            wln_diff_args = copy.deepcopy(args)
            wln_diff_args.wln_enc_node_dim = args.wln_enc_hidden_dim
            wln_diff_args.wln_enc_num_layers = 1
            self.wln_diff = WLNEncoder(wln_diff_args)
            self.final_transform = nn.Sequential(
                nn.Linear(
                    args.wln_enc_hidden_dim + 1
                    if self.add_scores_features
                    else args.wln_enc_hidden_dim,
                    args.wln_enc_hidden_dim,
                ),
                nn.ReLU(),
                nn.Linear(args.wln_enc_hidden_dim, 1),
            )

        elif args.use_chemprop_encoder:
            self.wln = DMPNNEncoder(args)  # WLN for mol representation
            wln_diff_args = copy.deepcopy(args)
            if args.model_name == 'wldn':
                wln_diff_args.chemprop_node_dim = args.chemprop_hidden_dim
            else:
                wln_diff_args.chemprop_edge_dim = args.chemprop_hidden_dim + 1
            wln_diff_args.chemprop_num_layers = 1
            self.wln_diff = DMPNNEncoder(wln_diff_args)
            self.final_transform = nn.Sequential(
                nn.Linear(
                    args.chemprop_hidden_dim + 1
                    if args.add_scores_to_edge_attr
                    else args.chemprop_hidden_dim,
                    args.chemprop_hidden_dim,
                ),
                # nn.BatchNorm1d(args.chemprop_hidden_dim),
                nn.ReLU(),
                nn.Linear(args.chemprop_hidden_dim, 1),
            )

        elif args.use_gat_encoder:
            # GAT
            self.wln = GAT(args)  # WLN for mol representation GAT(args)
            wln_diff_args = copy.deepcopy(args)
            wln_diff_args.gat_node_dim = args.gat_hidden_dim
            wln_diff_args.gat_num_layers = 1
            self.wln_diff = GAT(wln_diff_args)
            self.final_transform = nn.Sequential(
                nn.Linear(
                    args.gat_hidden_dim + 1
                    if self.add_scores_features
                    else args.gat_hidden_dim,
                    args.gat_hidden_dim,
                ),
                nn.ReLU(),
                nn.Linear(args.gat_hidden_dim, 1),
            )

        self.use_cache = (args.cache_path is not None) and (
            not args.load_wln_cache_in_dataset
        )
        if self.use_cache:
            self.cache = WLDN_Cache(os.path.join(args.cache_path), "pt")
        if self.args.test:
            assert not self.args.train

    def predict(self, batch, product_candidates_list, candidate_scores):
        predictions = []
        for idx, (candidates, scores) in enumerate(
            zip(product_candidates_list, candidate_scores)
        ):
            smiles_predictions = []
            # sort according to ranker score
            scores_indices = torch.argsort(scores.view(-1), descending=True)
            valid_candidate_combos = [
                candidates.candidate_bond_change[i] for i in scores_indices
            ]
            reactant_mol = Chem.MolFromSmiles(batch["reactants"].smiles[idx])
            for edits in valid_candidate_combos:
                edits_no_scores = [(x, y, t) for x, y, t, s in edits]
                smiles = robust_edit_mol(reactant_mol, edits_no_scores)
                # todo: smiles = set(sanitize_smiles_molvs(smiles) for smiles in smiles)
                if len(smiles) != 0:
                    smiles_predictions.append(smiles)
                else:
                    try:
                        Chem.Kekulize(reactant_mol)
                        smiles = robust_edit_mol(reactant_mol, edits_no_scores)
                        # todo: smiles = set(sanitize_smiles_molvs(smiles) for smiles in smiles)
                        smiles_predictions.append(smiles)
                    except Exception as e:
                        smiles_predictions.append([])
            predictions.append(smiles_predictions)

        return {"preds": predictions}

    def forward(self, batch):
        product_candidates_list = self.get_product_candidate_list(
            batch, batch["sample_id"]
        )

        reactant_node_feats = self.wln(batch["reactants"])[
            "node_features"
        ]  # N x D, where N is all the nodes in the batch
        dense_reactant_node_feats, mask = to_dense_batch(
            reactant_node_feats, batch=batch["reactants"].batch
        )  # B x max_batch_N x D
        candidate_scores = []
        for idx, product_candidates in enumerate(product_candidates_list):
            # get node features for candidate products
            product_candidates = product_candidates.to(reactant_node_feats.device)
            candidate_node_feats = self.wln(product_candidates)["node_features"]
            dense_candidate_node_feats, mask = to_dense_batch(
                candidate_node_feats, batch=product_candidates.batch
            )  # B x num_nodes x D

            num_nodes = dense_candidate_node_feats.shape[1]

            # compute difference vectors and replace the node features of the product graph with them
            difference_vectors = dense_candidate_node_feats - dense_reactant_node_feats[
                idx
            ][:num_nodes].unsqueeze(0)

            # undensify
            total_nodes = dense_candidate_node_feats.shape[0] * num_nodes
            difference_vectors = difference_vectors.view(total_nodes, -1)
            product_candidates.x = difference_vectors

            # apply a separate WLN to the difference graph
            wln_diff_output = self.wln_diff(product_candidates)
            diff_node_feats = wln_diff_output["node_features"]

            # compute the score for each candidate product
            # to dense
            diff_node_feats, _ = to_dense_batch(
                diff_node_feats, product_candidates.batch
            )  # num_candidates x max_num_nodes x D
            graph_feats = torch.sum(diff_node_feats, dim=-2)
            if self.add_scores_features:
                core_scores = [
                    sum(c[-1] for c in cand_changes)
                    for cand_changes in product_candidates.candidate_bond_change
                ]
                core_scores = torch.tensor(
                    core_scores, device=graph_feats.device
                ).unsqueeze(-1)
                score_order = torch.argsort(core_scores, dim=0)
                graph_feats = torch.concat([graph_feats, score_order], dim=-1)
            score = self.final_transform(graph_feats)
            if self.args.add_core_score:
                core_scores = [
                    sum(c[-1] for c in cand_changes)
                    for cand_changes in product_candidates.candidate_bond_change
                ]
                core_scores = torch.tensor(
                    core_scores, device=graph_feats.device
                ).unsqueeze(-1)
                score = score + torch.log_softmax(
                    core_scores, dim=0
                )  # torch.tensor(core_scores, device = score.device).unsqueeze(-1)
            candidate_scores.append(score)  # K x 1

        # ! PREDICT SMILES
        if self.args.predict:
            return self.predict(batch, product_candidates_list, candidate_scores)

        # note: dgl implementation adds reactivity score
        output = {
            "candidate_logit": candidate_scores,
            "product_candidates_list": product_candidates_list,
            # "s_uv": reactivity_output["s_uv"], # for debugging purposes
        }
        return output

    # seperate function because stays the same for different forward methods
    def get_product_candidate_list(self, batch, sample_ids):
        """
        Args:
            batch : collated samples from dataloader
            sample_ids: list of sample ids
        """
        if "product_candidates" in batch:
            return batch["product_candidates"]

        mode = (
            "test" if not self.args.train else "train"
        )  # this is used to get candidates, using robust_edit_mol when in predict later for actual smiles generation
        if self.use_cache:
            if not all(self.cache.exists(sid) for sid in sample_ids):
                with torch.no_grad():
                    reactivity_output = self.reactivity_net(
                        batch
                    )  # s_uv: N x N x 5, 'candidate_bond_changes', 'real_bond_changes'
                    if getattr(self.args, "mask_suv_scores", False):
                        reactivity_output["s_uv"][
                            ~batch["reactants"].mask.bool()
                        ] = -torch.inf

                # get candidate products as graph structures
                # each element in this list is a batch of candidate products (where each batch represents one reaction)
                product_candidates_list = generate_candidates_from_scores(
                    reactivity_output, batch, self.args, mode
                )
                [
                    self.cache.add(sid, product_candidates)
                    for sid, product_candidates in zip(
                        sample_ids, product_candidates_list
                    )
                ]
            else:
                product_candidates_list = [self.cache.get(sid) for sid in sample_ids]
        else:
            # each element in this list is a batch of candidate products (where each batch represents one reaction)
            with torch.no_grad():
                reactivity_output = self.reactivity_net(
                    batch
                )  # s_uv: N x N x 5, 'candidate_bond_changes', 'real_bond_changes'
                if getattr(self.args, "mask_suv_scores", False):
                    reactivity_output["s_uv"][
                        ~batch["reactants"].mask.bool()
                    ] = -torch.inf

            product_candidates_list = generate_candidates_from_scores(
                reactivity_output, batch, self.args, mode
            )

        return product_candidates_list

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        super(WLDN, WLDN).add_args(parser)
        DMPNNEncoder.add_args(parser)
        WLNEncoder.add_args(parser)
        parser.add_argument(
            "--num_candidate_bond_changes", type=int, default=20, help="Core size"
        )
        parser.add_argument(
            "--max_num_bond_changes", type=int, default=5, help="Combinations"
        )
        parser.add_argument(
            "--max_num_change_combos_per_reaction", type=int, default=500, help="cutoff"
        )
        parser.add_argument(
            "--reactivity_net_type",
            type=str,
            action=set_nox_type("model"),
            default="reaction_center_net",
            help="Type of reactivity net to use, mainly to init args",
        )
        parser.add_argument(
            "--reactivity_model_path",
            type=str,
            help="path to pretrained reaction center prediction model",
        )
        parser.add_argument(
            "--add_core_score",
            action="store_true",
            default=False,
            help="whether to add core score to ranking prediction",
        )
        #### Wln encoder
        parser.add_argument(
            "--use_wln_encoder",
            action="store_true",
            default=False,
            help="use wln implementation.",
        )
        parser.add_argument(
            "--use_chemprop_encoder",
            action="store_true",
            default=False,
            help="use wln implementation.",
        )
        parser.add_argument(
            "--use_gat_encoder",
            action="store_true",
            default=False,
            help="use gat implementation.",
        )
        parser.add_argument(
            "--add_scores_features",
            action="store_true",
            default=False,
            help="use gat implementation.",
        )
        parser.add_argument(
            "--mask_suv_scores",
            action="store_true",
            default=False,
            help="use mask to zero out scores for certain reactants.",
        )
        parser.add_argument(
            "--add_scores_to_edge_attr",
            action="store_true",
            default=False,
            help="use gat implementation.",
        )


##########################################################################################

######################################## Experimental ####################################


##########################################################################################
@register_object("ec_reaction_center_net", "model")
class ECReactivityCenterNet(ReactivityCenterNet):
    def __init__(self, args):
        super().__init__(args)

        embedding_dim = args.gat_hidden_dim // 4
        self.ec1 = nn.Embedding(len(args.ec_levels["1"]), embedding_dim)
        self.ec2 = nn.Embedding(len(args.ec_levels["2"]), embedding_dim)
        self.ec3 = nn.Embedding(len(args.ec_levels["3"]), embedding_dim)
        self.ec4 = nn.Embedding(len(args.ec_levels["4"]), embedding_dim)
        self.lin = nn.Linear(3 * args.gat_hidden_dim, args.gat_hidden_dim)

    def forward(self, batch):
        gat_output = self.gat_global_attention(
            batch["reactants"]
        )  # GAT + Global Attention over node features
        cs = gat_output["node_features"]
        c_tildes = gat_output["node_features_attn"]  # node contexts

        #
        ec_feats = torch.concat(
            [
                self.ec1(batch["ec1"]),
                self.ec2(batch["ec2"]),
                self.ec3(batch["ec3"]),
                self.ec4(batch["ec4"]),
            ],
            dim=-1,
        )
        repeat_counts = torch.bincount(batch["reactants"].batch)
        ec_feats = torch.repeat_interleave(ec_feats, repeat_counts, dim=0)

        c_final = self.lin(
            torch.cat([cs, c_tildes, ec_feats], dim=-1)
        )  # N x 3*hidden_dim -> N x hidden_dim

        s_uv = self.forward_helper(
            c_final,
            batch["reactants"]["edge_index_complete"],
            batch["reactants"]["edge_attr_complete"],
            batch["reactants"]["batch"],
        )

        # precompute for top k metric
        candidate_bond_changes = get_batch_candidate_bonds(
            batch["reaction"], s_uv.detach(), batch["reactants"].batch
        )
        # make bonds that are "4" -> "1.5"
        for i in range(len(candidate_bond_changes)):
            candidate_bond_changes[i] = [
                (elem[0], elem[1], 1.5, elem[3]) if elem[2] == 4 else elem
                for elem in candidate_bond_changes[i]
            ]

        batch_real_bond_changes = []
        for i in range(len(batch["reactants"]["bond_changes"])):
            reaction_real_bond_changes = []
            for elem in batch["reactants"]["bond_changes"][i]:
                reaction_real_bond_changes.append(tuple(elem))
            batch_real_bond_changes.append(reaction_real_bond_changes)

        assert len(candidate_bond_changes) == len(batch_real_bond_changes)

        return {
            "s_uv": s_uv,
            "candidate_bond_changes": candidate_bond_changes,
            "real_bond_changes": batch_real_bond_changes,
        }


@register_object("ec_wldn", "model")
class ECWLDN(WLDN):
    def __init__(self, args):
        super().__init__(args)
        embedding_dim = args.wln_enc_hidden_dim // 4
        self.ec1 = nn.Embedding(len(args.ec_levels["1"]), embedding_dim)
        self.ec2 = nn.Embedding(len(args.ec_levels["2"]), embedding_dim)
        self.ec3 = nn.Embedding(len(args.ec_levels["3"]), embedding_dim)
        self.ec4 = nn.Embedding(len(args.ec_levels["4"]), embedding_dim)

        self.final_transform = nn.Sequential(
            nn.Linear(
                2 * args.wln_enc_hidden_dim + 1
                if self.add_scores_features
                else 2 * args.wln_enc_hidden_dim,
                args.wln_enc_hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(args.wln_enc_hidden_dim, 1),
        )

    def forward(self, batch):
        product_candidates_list = self.get_product_candidate_list(
            batch, batch["sample_id"]
        )

        ec_feats = torch.concat(
            [
                self.ec1(batch["ec1"]),
                self.ec2(batch["ec2"]),
                self.ec3(batch["ec3"]),
                self.ec4(batch["ec4"]),
            ],
            dim=-1,
        )

        reactant_node_feats = self.wln(batch["reactants"])[
            "node_features"
        ]  # N x D, where N is all the nodes in the batch
        dense_reactant_node_feats, mask = to_dense_batch(
            reactant_node_feats, batch=batch["reactants"].batch
        )  # B x max_batch_N x D
        candidate_scores = []
        for idx, product_candidates in enumerate(product_candidates_list):
            # get node features for candidate products
            product_candidates = product_candidates.to(reactant_node_feats.device)
            candidate_node_feats = self.wln(product_candidates)["node_features"]
            dense_candidate_node_feats, mask = to_dense_batch(
                candidate_node_feats, batch=product_candidates.batch
            )  # B x num_nodes x D

            num_nodes = dense_candidate_node_feats.shape[1]

            # compute difference vectors and replace the node features of the product graph with them
            difference_vectors = dense_candidate_node_feats - dense_reactant_node_feats[
                idx
            ][:num_nodes].unsqueeze(0)

            # undensify
            total_nodes = dense_candidate_node_feats.shape[0] * num_nodes
            difference_vectors = difference_vectors.view(total_nodes, -1)
            product_candidates.x = difference_vectors

            # apply a separate WLN to the difference graph
            wln_diff_output = self.wln_diff(product_candidates)
            diff_node_feats = wln_diff_output["node_features"]

            # compute the score for each candidate product
            # to dense
            diff_node_feats, _ = to_dense_batch(
                diff_node_feats, product_candidates.batch
            )  # num_candidates x max_num_nodes x D
            graph_feats = torch.sum(diff_node_feats, dim=-2)

            graph_feats = torch.concat(
                [
                    graph_feats,
                    ec_feats[idx].unsqueeze(0).repeat(graph_feats.shape[0], 1),
                ],
                dim=-1,
            )

            if self.add_scores_features:
                core_scores = [
                    sum(c[-1] for c in cand_changes)
                    for cand_changes in product_candidates.candidate_bond_change
                ]
                core_scores = torch.tensor(
                    core_scores, device=graph_feats.device
                ).unsqueeze(-1)
                score_order = torch.argsort(core_scores, dim=0)
                graph_feats = torch.concat([graph_feats, score_order], dim=-1)
            score = self.final_transform(graph_feats)
            if self.args.add_core_score:
                core_scores = [
                    sum(c[-1] for c in cand_changes)
                    for cand_changes in product_candidates.candidate_bond_change
                ]
                core_scores = torch.tensor(
                    core_scores, device=graph_feats.device
                ).unsqueeze(-1)
                score = score + torch.log_softmax(
                    core_scores, dim=0
                )  # torch.tensor(core_scores, device = score.device).unsqueeze(-1)
            candidate_scores.append(score)  # K x 1

        # ! PREDICT SMILES
        if self.args.predict:
            return self.predict(batch, product_candidates_list, candidate_scores)

        # note: dgl implementation adds reactivity score
        output = {
            "candidate_logit": candidate_scores,
            "product_candidates_list": product_candidates_list,
            # "s_uv": reactivity_output["s_uv"], # for debugging purposes
        }
        return output


@register_object("chemprop_wldn", "model")
class ChempropWLDN(WLDN):
    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        super(ChempropWLDN, ChempropWLDN).add_args(parser)
        parser.add_argument(
            "--aggregate_over_edges",
            action="store_true",
            default=False,
            help="use gat implementation.",
        )
        parser.add_argument(
            "--add_scores_to_edge_attr",
            action="store_true",
            default=False,
            help="use gat implementation.",
        )

    def forward(self, batch):
        product_candidates_list = self.get_product_candidate_list(
            batch, batch["sample_id"]
        )

        reactant_edge_feats = self.wln(batch["reactants"])[
            "edge_features"
        ]  # N x D, where N is all the nodes in the batch
        dense_reactant_edge_feats = to_dense_adj(
            edge_index=batch["reactants"].edge_index,
            edge_attr=reactant_edge_feats,
            batch=batch["reactants"].batch,
        )
        graph_features = []
        candidate_scores = []
        for idx, product_candidates in enumerate(product_candidates_list):
            # get node features for candidate products
            product_candidates = product_candidates.to(reactant_edge_feats.device)
            candidate_edge_feats = self.wln(product_candidates)["edge_features"]

            dense_candidate_edge_feats = to_dense_adj(
                edge_index=product_candidates.edge_index,
                edge_attr=candidate_edge_feats,
                batch=product_candidates.batch,
            )

            num_nodes = dense_candidate_edge_feats.shape[1]

            # add edge vectors
            sum_vectors = dense_candidate_edge_feats + dense_reactant_edge_feats[idx][
                :num_nodes, :num_nodes
            ].unsqueeze(
                0
            )  # B, N, N, D

            # get matrix for scores consistent with edge_attr
            score_features = torch.zeros_like(dense_candidate_edge_feats)[..., :1]
            for cid, cand_changes in enumerate(
                product_candidates.candidate_bond_change
            ):
                for x, y, t, s in cand_changes:
                    score_features[cid, x, y] = s
                    score_features[cid, y, x] = s
            sum_vectors = torch.concat([sum_vectors, score_features], dim=-1)

            # undensify
            flat_sum_vectors = sum_vectors.sum(-1)
            new_edge_indices = [dense_to_sparse(E)[0] for E in flat_sum_vectors]
            new_edge_attr = torch.vstack(
                [sum_vectors[i, e[0], e[1]] for i, e in enumerate(new_edge_indices)]
            )
            new_edge_index, _ = dense_to_sparse(flat_sum_vectors)
            product_candidates.edge_attr = new_edge_attr
            product_candidates.edge_index = new_edge_index
            score_features_edge_attr = new_edge_attr[..., -1:]

            # apply a separate WLN to the difference graph
            wln_diff_output = self.wln_diff(product_candidates)
            sum_node_feats = wln_diff_output["node_features"]

            if self.args.aggregate_over_edges:
                # compute the score for each candidate product
                sum_edge_feats = wln_diff_output["edge_features"]
                edge_batch = product_candidates.batch[new_edge_index[0]]
                if self.args.add_scores_to_edge_attr:
                    sum_edge_feats = torch.concat(
                        [sum_edge_feats, score_features_edge_attr], dim=-1
                    )
                score_per_edge = self.final_transform(sum_edge_feats)
                score = scatter(score_per_edge, edge_batch, dim=0, reduce="mean")
            else:
                sum_node_feats, _ = to_dense_batch(
                    sum_node_feats, product_candidates.batch
                )  # num_candidates x max_num_nodes x D
                graph_feats = torch.sum(sum_node_feats, dim=-2)
                score = self.final_transform(graph_feats)
                graph_features.append(graph_feats)

            if self.args.add_core_score:
                core_scores = [
                    sum(c[-1] for c in cand_changes)
                    for cand_changes in product_candidates.candidate_bond_change
                ]
                core_scores = torch.tensor(core_scores, device=score.device).unsqueeze(
                    -1
                )
                score = score + torch.log_softmax(
                    core_scores, dim=0
                )  # torch.tensor(core_scores, device = score.device).unsqueeze(-1)
            candidate_scores.append(score)  # K x 1

        # ! PREDICT SMILES
        if self.args.predict:
            return self.predict(batch, product_candidates_list, candidate_scores)

        # note: dgl implementation adds reactivity score
        output = {
            "candidate_logit": candidate_scores,
            "product_candidates_list": product_candidates_list,
            "hidden": graph_features,
            # "s_uv": reactivity_output["s_uv"], # for debugging purposes
        }
        return output


@register_object("chemprop_cgr", "model")
class ChempropCGR(WLDN):
    def __init__(self, args):
        super().__init__(args)
        self.final_transform = nn.Sequential(
            nn.Linear(
                args.chemprop_hidden_dim + 1
                if args.add_scores_to_edge_attr
                else args.chemprop_hidden_dim,
                args.chemprop_hidden_dim,
            ),
            nn.LayerNorm(args.chemprop_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.chemprop_hidden_dim, 1),
        )

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        super(ChempropCGR, ChempropCGR).add_args(parser)
        parser.add_argument(
            "--aggregate_over_edges",
            action="store_true",
            default=False,
            help="use gat implementation.",
        )
        parser.add_argument(
            "--add_scores_to_edge_attr",
            action="store_true",
            default=False,
            help="use gat implementation.",
        )

    def forward(self, batch):
        product_candidates_list = self.get_product_candidate_list(
            batch, batch["sample_id"]
        )

        dense_reactant_node_feats, _ = to_dense_batch(
            batch["reactants"].x, batch["reactants"].batch
        )  # batch size, max num nodes, feature dimension

        dense_reactant_edge_feats = to_dense_adj(
            edge_index=batch["reactants"].edge_index,
            edge_attr=batch["reactants"].edge_attr
            + 1,  # ! add 1 since encoding no edge here
            batch=batch["reactants"].batch,
        )

        graph_features = []
        candidate_scores = []
        for idx, product_candidates in enumerate(product_candidates_list):
            product_candidates = product_candidates.to(dense_reactant_edge_feats.device)

            dense_candidate_edge_feats = to_dense_adj(
                edge_index=product_candidates.edge_index,
                edge_attr=product_candidates.edge_attr
                + 1,  # ! add 1 since encoding no edge here
                batch=product_candidates.batch,
            )

            # node features
            x, _ = to_dense_batch(
                product_candidates.x, product_candidates.batch
            )  # batch size, num nodes, feature dimension
            ncands, num_nodes, _ = x.shape
            cgr_nodes = torch.cat(
                [
                    dense_reactant_node_feats[idx][:num_nodes]
                    .unsqueeze(0)
                    .repeat(ncands, 1, 1),
                    x,
                ],
                dim=-1,
            )
            product_candidates.x = cgr_nodes.view(-1, cgr_nodes.shape[-1])

            # edge features
            cgr_attr = torch.cat(
                [
                    dense_reactant_edge_feats[idx][:num_nodes, :num_nodes]
                    .unsqueeze(0)
                    .repeat(ncands, 1, 1, 1),
                    dense_candidate_edge_feats,
                ],
                dim=-1,
            )  # B, N, N, D

            # undensify
            flat_sum_vectors = cgr_attr.sum(-1)
            new_edge_indices = [dense_to_sparse(E)[0] for E in flat_sum_vectors]
            new_edge_attr = torch.vstack(
                [cgr_attr[i, e[0], e[1]] for i, e in enumerate(new_edge_indices)]
            )
            new_edge_index, _ = dense_to_sparse(flat_sum_vectors)
            product_candidates.edge_attr = new_edge_attr
            product_candidates.edge_index = new_edge_index

            # apply a separate WLN to the difference graph
            wln_diff_output = self.wln(product_candidates)
            sum_node_feats = wln_diff_output["node_features"]

            sum_node_feats, _ = to_dense_batch(
                sum_node_feats, product_candidates.batch
            )  # num_candidates x max_num_nodes x D
            graph_feats = torch.sum(sum_node_feats, dim=-2)
            score = self.final_transform(graph_feats)
            graph_features.append(graph_feats)

            if self.args.add_core_score:
                core_scores = [
                    sum(c[-1] for c in cand_changes)
                    for cand_changes in product_candidates.candidate_bond_change
                ]
                core_scores = torch.tensor(core_scores, device=score.device).unsqueeze(
                    -1
                )
                score = score + torch.log_softmax(
                    core_scores, dim=0
                )  # torch.tensor(core_scores, device = score.device).unsqueeze(-1)
            candidate_scores.append(score)  # K x 1

        # ! PREDICT SMILES
        if self.args.predict:
            return self.predict(batch, product_candidates_list, candidate_scores)

        # note: dgl implementation adds reactivity score
        output = {
            "candidate_logit": candidate_scores,
            "product_candidates_list": product_candidates_list,
            "hidden": graph_features,
            # "s_uv": reactivity_output["s_uv"], # for debugging purposes
        }
        return output


from transformers import (
    BertConfig,
    BertModel,
    AutoTokenizer,
    EsmModel,
    BertTokenizer,
    BertLMHeadModel,
    BertForMaskedLM,
)
from nox.utils.smiles import remove_atom_maps, standardize_reaction, tokenize_smiles
from typing import Union, Tuple, Any, List, Dict, Optional


@register_object("esm_reaction_center_net", "model")
class ESMReactivityCenterNet(ReactivityCenterNet):
    def __init__(self, args):
        super().__init__(args)
        self.esm_tokenizer = AutoTokenizer.from_pretrained(args.esm_model_version)
        self.esm_model = EsmModel.from_pretrained(args.esm_model_version)
        self.freeze_encoder = args.freeze_encoder
        if self.freeze_encoder:
            self.esm_model.requires_grad_(False)
        self.mha = nn.MultiheadAttention(
            embed_dim=self.esm_model.config.hidden_size,
            num_heads=args.num_heads,
            batch_first=True,
        )
        self.lin = nn.Linear(3 * args.gat_hidden_dim, args.gat_hidden_dim)

        self.predict_ecs = getattr(args, "predict_ecs", False)
        if self.predict_ecs:
            ec_args = copy.deepcopy(args)
            ec_args.num_classes = len(args.ec_levels["4"].values())
            self.ec_classifier = get_object(args.ec_classifier, "model")(ec_args)

    def encode_sequence(self, batch):
        encoder_input_ids = self.esm_tokenizer(
            batch["sequence"],
            padding="longest",
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        for k, v in encoder_input_ids.items():
            encoder_input_ids[k] = v.to(self.lin.weight.device)

        if self.freeze_encoder:
            self.esm_model.requires_grad_(False)
            with torch.no_grad():
                encoder_outputs = self.esm_model(
                    input_ids=encoder_input_ids["input_ids"],
                    attention_mask=encoder_input_ids["attention_mask"],
                )
        else:
            encoder_outputs = self.esm_model(
                input_ids=encoder_input_ids["input_ids"],
                attention_mask=encoder_input_ids["attention_mask"],
            )
        encoder_hidden_states = encoder_outputs["last_hidden_state"]
        encoder_pooled_states = encoder_outputs["pooler_output"].unsqueeze(1)
        mask = encoder_input_ids["attention_mask"]  # [:,0:1]
        return encoder_hidden_states, encoder_pooled_states, mask

    def forward(self, batch):
        if all([isinstance(l, Data) for l in batch["reactants"]]):
            batch["reactants"] = Batch.from_data_list(batch["reactants"])
        gat_output = self.gat_global_attention(
            batch["reactants"]
        )  # GAT + Global Attention over node features
        cs = gat_output["node_features"]
        c_tildes = gat_output["node_features_attn"]  # node contexts

        # protein
        protein_feats, protein_pooled_feats, protein_mask = self.encode_sequence(batch)
        dense_cs, cs_mask = to_dense_batch(cs, batch["reactants"].batch)
        c_protein, _ = self.mha(  # batch_size, max_num_nodes (reactants), hidden dim
            query=dense_cs,
            key=protein_feats,
            value=protein_feats,
            key_padding_mask=protein_mask.float(),
        )
        c_protein_pool = c_protein.sum(1)
        c_protein = c_protein[cs_mask]  # num nodes, dim

        c_final = self.lin(
            torch.cat([cs, c_tildes, c_protein], dim=-1)
        )  # N x 3*hidden_dim -> N x hidden_dim

        s_uv = self.forward_helper(
            c_final,
            batch["reactants"]["edge_index_complete"],
            batch["reactants"]["edge_attr_complete"],
            batch["reactants"]["batch"],
        )

        # precompute for top k metric
        candidate_bond_changes = get_batch_candidate_bonds(
            batch["reaction"], s_uv.detach(), batch["reactants"].batch
        )
        # make bonds that are "4" -> "1.5"
        for i in range(len(candidate_bond_changes)):
            candidate_bond_changes[i] = [
                (elem[0], elem[1], 1.5, elem[3]) if elem[2] == 4 else elem
                for elem in candidate_bond_changes[i]
            ]

        batch_real_bond_changes = []
        for i in range(len(batch["reactants"]["bond_changes"])):
            reaction_real_bond_changes = []
            for elem in batch["reactants"]["bond_changes"][i]:
                reaction_real_bond_changes.append(tuple(elem))
            batch_real_bond_changes.append(reaction_real_bond_changes)

        assert len(candidate_bond_changes) == len(batch_real_bond_changes)

        # predict ecs
        ec_logits = (
            self.ec_classifier({"x": c_protein_pool})["logit"]
            if self.predict_ecs
            else None
        )

        return {
            "s_uv": s_uv,
            "candidate_bond_changes": candidate_bond_changes,
            "real_bond_changes": batch_real_bond_changes,
            "ec_logits": ec_logits,
            "c_final": c_final,
        }

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        super(ESMReactivityCenterNet, ESMReactivityCenterNet).add_args(parser)
        parser.add_argument(
            "--esm_model_version",
            type=str,
            default=None,
            help="which version of ESM to use",
        )
        parser.add_argument(
            "--num_heads",
            type=int,
            default=8,
            help="maximum length allowed for the input sequence",
        )
        parser.add_argument(
            "--freeze_encoder",
            action="store_true",
            default=False,
            help="whether use model as pre-trained encoder and not update weights",
        )
        parser.add_argument(
            "--predict_ecs",
            action="store_true",
            default=False,
            help="do ec classification.",
        )
        parser.add_argument(
            "--ec_classifier",
            type=str,
            action=set_nox_type("model"),
            default="mlp_classifier",
            help="mlp",
        )


from nox.models.egnn import EGNN_Sparse_Network


@register_object("egnn_reaction_center_net", "model")
class EGNNReactivityCenterNet(ESMReactivityCenterNet):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.egnn = get_object(args.egnn_name, "model")(args)

        self.mha = nn.MultiheadAttention(
            embed_dim=args.gat_hidden_dim,
            num_heads=args.num_heads,
            batch_first=True,
        )

        self.prot_proj = nn.Linear(args.protein_dim, args.gat_hidden_dim)

        del self.esm_model
        del self.esm_tokenizer

    def encode_sequence(self, batch):
        feats, coors = self.egnn(batch)
        try:
            batch_idxs = batch["graph"]["receptor"].batch
        except:
            batch_idxs = batch["receptor"].batch

        encoder_pooled_states = scatter(
            feats,
            batch_idxs,
            dim=0,
            reduce=self.args.pool_type,
        )

        encoder_hidden_states, mask = to_dense_batch(feats, batch_idxs)

        # if node dim in EGNN is not the same as the gat dim then project down
        # likely due to ESM versions
        if self.args.protein_dim != self.args.gat_hidden_dim:
            encoder_pooled_states = self.prot_proj(
                encoder_pooled_states
            )  # B x prot_dim -> B x gat_hidden_dim
            encoder_hidden_states = self.prot_proj(
                encoder_hidden_states
            )  # B x max_num_nodes x prot_dim -> B x max_num_nodes x gat_hidden_dim

        return encoder_hidden_states, encoder_pooled_states, mask

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        super(EGNNReactivityCenterNet, EGNNReactivityCenterNet).add_args(parser)
        parser.add_argument(
            "--egnn_name",
            type=str,
            action=set_nox_type("model"),
            default="egnn_sparse_network",
            help="mlp",
        )


@register_object("esm_wldn_cgr", "model")
class ChempropESMCGR(WLDN):
    def __init__(self, args):
        super().__init__(args)

        self.esm_tokenizer = AutoTokenizer.from_pretrained(args.wln_esm_model_version)
        self.esm_model = EsmModel.from_pretrained(args.wln_esm_model_version)
        self.freeze_encoder = args.wln_freeze_encoder
        if self.freeze_encoder:
            self.esm_model.requires_grad_(False)

        self.mha = nn.MultiheadAttention(
            embed_dim=self.esm_model.config.hidden_size,
            num_heads=args.wln_num_heads,
            batch_first=True,
        )

        self.final_transform = nn.Sequential(
            nn.Linear(
                args.chemprop_hidden_dim * 2 + 1
                if args.add_scores_to_edge_attr
                else args.chemprop_hidden_dim * 2,
                args.chemprop_hidden_dim,
            ),
            nn.LayerNorm(args.chemprop_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.chemprop_hidden_dim, 1),
        )

        self.predict_ecs = getattr(args, "wldn_predict_ecs", False)
        if self.predict_ecs:
            ec_args = copy.deepcopy(args)
            ec_args.num_classes = len(args.ec_levels["4"].values())
            self.ec_classifier = get_object(args.wldn_ec_classifier, "model")(ec_args)

    def encode_sequence(self, batch):
        encoder_input_ids = self.esm_tokenizer(
            batch["sequence"],
            padding="longest",
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        for k, v in encoder_input_ids.items():
            encoder_input_ids[k] = v.to(self.mha.out_proj.weight.device)

        if self.freeze_encoder:
            self.esm_model.requires_grad_(False)
            with torch.no_grad():
                encoder_outputs = self.esm_model(
                    input_ids=encoder_input_ids["input_ids"],
                    attention_mask=encoder_input_ids["attention_mask"],
                )
        else:
            encoder_outputs = self.esm_model(
                input_ids=encoder_input_ids["input_ids"],
                attention_mask=encoder_input_ids["attention_mask"],
            )
        encoder_hidden_states = encoder_outputs["last_hidden_state"]
        protein_pooled_feats = encoder_outputs["pooler_output"].unsqueeze(1)
        mask = encoder_input_ids["attention_mask"]  # [:,0:1]
        return encoder_hidden_states, protein_pooled_feats, mask

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        super(ChempropESMCGR, ChempropESMCGR).add_args(parser)
        parser.add_argument(
            "--aggregate_over_edges",
            action="store_true",
            default=False,
            help="use gat implementation.",
        )
        parser.add_argument(
            "--add_scores_to_edge_attr",
            action="store_true",
            default=False,
            help="use gat implementation.",
        )
        parser.add_argument(
            "--wln_esm_model_version",
            type=str,
            default=None,
            help="which version of ESM to use",
        )
        parser.add_argument(
            "--wln_num_heads",
            type=int,
            default=8,
            help="maximum length allowed for the input sequence",
        )
        parser.add_argument(
            "--wln_freeze_encoder",
            action="store_true",
            default=False,
            help="whether use model as pre-trained encoder and not update weights",
        )
        parser.add_argument(
            "--wldn_predict_ecs",
            action="store_true",
            default=False,
            help="do ec classification.",
        )
        parser.add_argument(
            "--wldn_ec_classifier",
            type=str,
            action=set_nox_type("model"),
            default="mlp_classifier",
            help="mlp",
        )

    def forward(self, batch):
        protein_feats, protein_pooled_feats, protein_mask = self.encode_sequence(batch)

        product_candidates_list = self.get_product_candidate_list(
            batch, batch["sample_id"]
        )

        dense_reactant_node_feats, _ = to_dense_batch(
            batch["reactants"].x, batch["reactants"].batch
        )  # batch size, max num nodes, feature dimension

        dense_reactant_edge_feats = to_dense_adj(
            edge_index=batch["reactants"].edge_index,
            edge_attr=batch["reactants"].edge_attr
            + 1,  # ! add 1 since encoding no edge here
            batch=batch["reactants"].batch,
        )

        graph_features = []
        candidate_scores = []
        for idx, product_candidates in enumerate(product_candidates_list):
            product_candidates = product_candidates.to(dense_reactant_edge_feats.device)

            dense_candidate_edge_feats = to_dense_adj(
                edge_index=product_candidates.edge_index,
                edge_attr=product_candidates.edge_attr
                + 1,  # ! add 1 since encoding no edge here
                batch=product_candidates.batch,
            )

            # node features
            x, _ = to_dense_batch(
                product_candidates.x, product_candidates.batch
            )  # batch size, num nodes, feature dimension
            ncands, num_nodes, _ = x.shape
            cgr_nodes = torch.cat(
                [
                    dense_reactant_node_feats[idx][:num_nodes]
                    .unsqueeze(0)
                    .repeat(ncands, 1, 1),
                    x,
                ],
                dim=-1,
            )
            product_candidates.x = cgr_nodes.view(-1, cgr_nodes.shape[-1])

            # edge features
            cgr_attr = torch.cat(
                [
                    dense_reactant_edge_feats[idx][:num_nodes, :num_nodes]
                    .unsqueeze(0)
                    .repeat(ncands, 1, 1, 1),
                    dense_candidate_edge_feats,
                ],
                dim=-1,
            )  # B, N, N, D

            # undensify
            flat_sum_vectors = cgr_attr.sum(-1)
            new_edge_indices = [dense_to_sparse(E)[0] for E in flat_sum_vectors]
            new_edge_attr = torch.vstack(
                [cgr_attr[i, e[0], e[1]] for i, e in enumerate(new_edge_indices)]
            )
            new_edge_index, _ = dense_to_sparse(flat_sum_vectors)
            product_candidates.edge_attr = new_edge_attr
            product_candidates.edge_index = new_edge_index

            # apply a separate WLN to the difference graph
            wln_diff_output = self.wln(product_candidates)
            sum_node_feats = wln_diff_output["node_features"]

            sum_node_feats, sum_node_feats_mask = to_dense_batch(
                sum_node_feats, product_candidates.batch
            )  # num_candidates x max_num_nodes x D
            protein_rxn_feats = (
                protein_feats[idx][: protein_mask[idx].sum()]
                .unsqueeze(0)
                .repeat(sum_node_feats.shape[0], 1, 1)
            )

            c_protein = []
            for minibatch in range(0, len(sum_node_feats), 32):
                c_, _ = self.mha(  # batch_size, max_num_nodes (reactants), hidden dim
                    query=sum_node_feats[minibatch : (minibatch + 32)],
                    key=protein_rxn_feats[minibatch : (minibatch + 32)],
                    value=protein_rxn_feats[minibatch : (minibatch + 32)],
                )
                c_protein.append(c_)

            c_protein = torch.vstack(c_protein)

            # c_protein, _ = self.mha( # batch_size, max_num_nodes (reactants), hidden dim
            #     query=sum_node_feats,
            #     key=protein_rxn_feats,
            #     value=protein_rxn_feats
            # )
            sum_node_feats = torch.cat([sum_node_feats, c_protein], dim=-1)
            graph_feats = torch.sum(sum_node_feats, dim=-2)
            score = self.final_transform(graph_feats)
            graph_features.append(graph_feats)

            if self.args.add_core_score:
                core_scores = [
                    sum(c[-1] for c in cand_changes)
                    for cand_changes in product_candidates.candidate_bond_change
                ]
                core_scores = torch.tensor(core_scores, device=score.device).unsqueeze(
                    -1
                )
                score = score + torch.log_softmax(
                    core_scores, dim=0
                )  # torch.tensor(core_scores, device = score.device).unsqueeze(-1)
            candidate_scores.append(score)  # K x 1

        # ! PREDICT SMILES
        if self.args.predict:
            return self.predict(batch, product_candidates_list, candidate_scores)

        # predict ecs
        true_candidate_product = torch.stack([g[0] for g in graph_features])
        ec_logits = (
            self.ec_classifier({"x": true_candidate_product})["logit"]
            if self.predict_ecs
            else None
        )

        # note: dgl implementation adds reactivity score
        output = {
            "candidate_logit": candidate_scores,
            "product_candidates_list": product_candidates_list,
            "hidden": graph_features,
            "ec_logits": ec_logits,
        }
        return output


@register_object("esm_wldn_cgr_late_fusion", "model")
class ChempropESMCGRLate(WLDN):
    def __init__(self, args):
        super().__init__(args)

        self.esm_tokenizer = AutoTokenizer.from_pretrained(args.wln_esm_model_version)
        self.esm_model = EsmModel.from_pretrained(args.wln_esm_model_version)
        self.freeze_encoder = args.wln_freeze_encoder
        if self.freeze_encoder:
            self.esm_model.requires_grad_(False)

        self.final_transform = nn.Sequential(
            nn.Linear(
                args.chemprop_hidden_dim * 2 + 1
                if args.add_scores_to_edge_attr
                else args.chemprop_hidden_dim * 2,
                args.chemprop_hidden_dim,
            ),
            nn.LayerNorm(args.chemprop_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.chemprop_hidden_dim, 1),
        )

        self.predict_ecs = getattr(args, "wldn_predict_ecs", False)
        if self.predict_ecs:
            ec_args = copy.deepcopy(args)
            ec_args.num_classes = len(args.ec_levels["4"].values())
            self.ec_classifier = get_object(args.wldn_ec_classifier, "model")(ec_args)

    def encode_sequence(self, batch):
        encoder_input_ids = self.esm_tokenizer(
            batch["sequence"],
            padding="longest",
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        for k, v in encoder_input_ids.items():
            encoder_input_ids[k] = v.to(self.esm_model.device)

        if self.freeze_encoder:
            self.esm_model.requires_grad_(False)
            with torch.no_grad():
                encoder_outputs = self.esm_model(
                    input_ids=encoder_input_ids["input_ids"],
                    attention_mask=encoder_input_ids["attention_mask"],
                )
        else:
            encoder_outputs = self.esm_model(
                input_ids=encoder_input_ids["input_ids"],
                attention_mask=encoder_input_ids["attention_mask"],
            )
        encoder_hidden_states = encoder_outputs["last_hidden_state"]
        protein_pooled_feats = encoder_outputs["pooler_output"].unsqueeze(1)
        mask = encoder_input_ids["attention_mask"]  # [:,0:1]
        return encoder_hidden_states, protein_pooled_feats, mask

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        super(ChempropESMCGRLate, ChempropESMCGRLate).add_args(parser)
        parser.add_argument(
            "--aggregate_over_edges",
            action="store_true",
            default=False,
            help="use gat implementation.",
        )
        parser.add_argument(
            "--add_scores_to_edge_attr",
            action="store_true",
            default=False,
            help="use gat implementation.",
        )
        parser.add_argument(
            "--wln_esm_model_version",
            type=str,
            default=None,
            help="which version of ESM to use",
        )
        parser.add_argument(
            "--wln_num_heads",
            type=int,
            default=8,
            help="maximum length allowed for the input sequence",
        )
        parser.add_argument(
            "--wln_freeze_encoder",
            action="store_true",
            default=False,
            help="whether use model as pre-trained encoder and not update weights",
        )
        parser.add_argument(
            "--wldn_predict_ecs",
            action="store_true",
            default=False,
            help="do ec classification.",
        )
        parser.add_argument(
            "--wldn_ec_classifier",
            type=str,
            action=set_nox_type("model"),
            default="mlp_classifier",
            help="mlp",
        )

    def forward(self, batch):
        protein_feats, protein_pooled_feats, protein_mask = self.encode_sequence(batch)

        product_candidates_list = self.get_product_candidate_list(
            batch, batch["sample_id"]
        )

        dense_reactant_node_feats, _ = to_dense_batch(
            batch["reactants"].x, batch["reactants"].batch
        )  # batch size, max num nodes, feature dimension

        dense_reactant_edge_feats = to_dense_adj(
            edge_index=batch["reactants"].edge_index,
            edge_attr=batch["reactants"].edge_attr
            + 1,  # ! add 1 since encoding no edge here
            batch=batch["reactants"].batch,
        )

        graph_features = []
        candidate_scores = []
        for idx, product_candidates in enumerate(product_candidates_list):
            product_candidates = product_candidates.to(dense_reactant_edge_feats.device)

            dense_candidate_edge_feats = to_dense_adj(
                edge_index=product_candidates.edge_index,
                edge_attr=product_candidates.edge_attr
                + 1,  # ! add 1 since encoding no edge here
                batch=product_candidates.batch,
            )

            # node features
            x, _ = to_dense_batch(
                product_candidates.x, product_candidates.batch
            )  # batch size, num nodes, feature dimension
            ncands, num_nodes, _ = x.shape
            cgr_nodes = torch.cat(
                [
                    dense_reactant_node_feats[idx][:num_nodes]
                    .unsqueeze(0)
                    .repeat(ncands, 1, 1),
                    x,
                ],
                dim=-1,
            )
            product_candidates.x = cgr_nodes.view(-1, cgr_nodes.shape[-1])

            # edge features
            cgr_attr = torch.cat(
                [
                    dense_reactant_edge_feats[idx][:num_nodes, :num_nodes]
                    .unsqueeze(0)
                    .repeat(ncands, 1, 1, 1),
                    dense_candidate_edge_feats,
                ],
                dim=-1,
            )  # B, N, N, D

            # undensify
            flat_sum_vectors = cgr_attr.sum(-1)
            new_edge_indices = [dense_to_sparse(E)[0] for E in flat_sum_vectors]
            new_edge_attr = torch.vstack(
                [cgr_attr[i, e[0], e[1]] for i, e in enumerate(new_edge_indices)]
            )
            new_edge_index, _ = dense_to_sparse(flat_sum_vectors)
            product_candidates.edge_attr = new_edge_attr
            product_candidates.edge_index = new_edge_index

            # apply a separate WLN to the difference graph
            wln_diff_output = self.wln(product_candidates)
            sum_node_feats = wln_diff_output["node_features"]

            sum_node_feats, sum_node_feats_mask = to_dense_batch(
                sum_node_feats, product_candidates.batch
            )  # num_candidates x max_num_nodes x D
            protein_rxn_feats = (
                protein_feats[idx][1 : (protein_mask[idx].sum() - 1)]
                .mean(0)
                .unsqueeze(0)
                .repeat(sum_node_feats.shape[0], 1)
            )

            graph_feats = torch.sum(sum_node_feats, dim=-2)
            graph_feats = torch.cat([graph_feats, protein_rxn_feats], dim=-1)
            score = self.final_transform(graph_feats)
            graph_features.append(graph_feats)

            if self.args.add_core_score:
                core_scores = [
                    sum(c[-1] for c in cand_changes)
                    for cand_changes in product_candidates.candidate_bond_change
                ]
                core_scores = torch.tensor(core_scores, device=score.device).unsqueeze(
                    -1
                )
                score = score + torch.log_softmax(
                    core_scores, dim=0
                )  # torch.tensor(core_scores, device = score.device).unsqueeze(-1)
            candidate_scores.append(score)  # K x 1

        # ! PREDICT SMILES
        if self.args.predict:
            return self.predict(batch, product_candidates_list, candidate_scores)

        # predict ecs
        true_candidate_product = torch.vstack([g[0] for g in graph_features])
        ec_logits = (
            self.ec_classifier({"x": true_candidate_product})["logit"]
            if self.predict_ecs
            else None
        )

        # note: dgl implementation adds reactivity score
        output = {
            "candidate_logit": candidate_scores,
            "product_candidates_list": product_candidates_list,
            "hidden": graph_features,
            "ec_logits": ec_logits,
        }
        return output


@register_object("transformer_reaction_ranker", "model")
class TransformerRanker(WLDN):
    def __init__(self, args):
        super().__init__(args)
        self.tokenizer = self.init_tokenizer(args)
        econfig = self.get_transformer_config(args)
        econfig.is_decoder = False
        if args.mlm_model_path:
            state_dict = torch.load(args.mlm_model_path)
            econfig = self.get_transformer_config(
                state_dict["hyper_parameters"]["args"]
            )
            self.model = BertForMaskedLM(econfig)
            self.model.load_state_dict(
                {
                    k[len("model.model.") :]: v
                    for k, v in state_dict["state_dict"].items()
                    if k.startswith("model.model")
                }
            )
        else:
            self.model = BertForMaskedLM(econfig)
        self.config = self.model.config
        self.args = args
        self.register_buffer("token_type_ids", torch.zeros(1, dtype=torch.long))

        self.final_transform = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, 1),
        )  # need to overwrite because output is different size

    @staticmethod
    def init_tokenizer(args):
        return BertTokenizer(
            vocab_file=args.vocab_path,
            do_lower_case=False,
            do_basic_tokenize=False,
            # sep_token=".",
            padding=True,
            truncation=True,
            model_max_length=args.max_seq_len,
            additional_special_tokens=[">>"],
            cls_token="[CLS]",
            eos_token="[EOS]",
        )

    def get_transformer_config(self, args):
        config = BertConfig(
            vocab_size=self.tokenizer.vocab_size,
            max_position_embeddings=args.max_seq_len,
            type_vocab_size=2,
            output_hidden_states=True,
            output_attentions=True,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            num_attention_heads=args.num_heads,
            num_hidden_layers=args.num_hidden_layers,
        )
        return config

    @staticmethod
    def tokenize(candidate_reactions, tokenizer, args):
        x = [standardize_reaction(r) for r in candidate_reactions]
        x = [tokenize_smiles(r, return_as_str=True) for r in x]

        if args.use_cls_token:
            # add [CLS] and [EOS] tokens
            x = [f"{tokenizer.cls_token} {r} {tokenizer.eos_token}" for r in x]

        # tokenize str characters into tensor of indices with corresponding masks
        tokenized_inputs = tokenizer(
            x,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )

        # get mask for special tokens that are not masked in MLM (return_special_tokens_mask=True doesn't work for additional special tokens)
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(
                toks, None, already_has_special_tokens=True
            )
            for toks in tokenized_inputs["input_ids"]
        ]

        tokenized_inputs["special_tokens_mask"] = torch.tensor(
            special_tokens_mask, dtype=torch.int64
        )

        return tokenized_inputs

    def forward(self, batch):
        reactant_smiles = [
            remove_atom_maps(s, sanitize=False) for s in batch["reactants"].smiles
        ]

        product_candidates_list = self.get_product_candidate_list(
            batch, batch["row_id"]
        )

        candidate_scores = []
        inputs_for_mlm = []
        for idx, product_candidates in enumerate(product_candidates_list):
            # get node features for candidate products
            candidate_smiles = [
                remove_atom_maps(s, sanitize=False) for s in product_candidates.smiles
            ]
            candidate_reactions = [
                "{}>>{}".format(reactant_smiles[idx], c) for c in candidate_smiles
            ]

            tokenized_inputs = self.tokenize(
                candidate_reactions, self.tokenizer, self.args
            )
            for k, v in tokenized_inputs.items():
                tokenized_inputs[k] = v.to(self.token_type_ids.device)

            batched_candidates_scores = []
            for j in range(0, len(candidate_reactions), 32):
                outputs = self.model.bert(
                    input_ids=tokenized_inputs["input_ids"][j : (j + 32)],
                    attention_mask=tokenized_inputs["attention_mask"][j : (j + 32)],
                    token_type_ids=None,
                )

                sequence_output = outputs[0]
                score = self.final_transform(sequence_output[:, 0])
                batched_candidates_scores.append(score)

            score = torch.vstack(batched_candidates_scores)
            if self.args.add_core_score:
                core_scores = [
                    sum(c[-1] for c in cand_changes)
                    for cand_changes in product_candidates.candidate_bond_change
                ]
                score = score + torch.tensor(
                    core_scores, device=score.device
                ).unsqueeze(-1)
            candidate_scores.append(score)  # K x 1

            if self.args.do_masked_language_model:
                inputs_for_mlm.append(candidate_reactions[0])

        output = {
            "candidate_logit": candidate_scores,
            "product_candidates_list": product_candidates_list,
        }

        if self.args.do_masked_language_model:
            tokenized_inputs = self.tokenize(inputs_for_mlm, self.tokenizer, self.args)

            for k, v in tokenized_inputs.items():
                tokenized_inputs[k] = v.to(self.token_type_ids.device)

            special_tokens_mask = tokenized_inputs.pop("special_tokens_mask", None)
            # mask tokens according to probability
            (
                tokenized_inputs["input_ids"],
                tokenized_inputs["labels"],
            ) = self.torch_mask_tokens(
                tokenized_inputs["input_ids"], special_tokens_mask=special_tokens_mask
            )
            # do MLM
            mlm_scores = self.model(**tokenized_inputs)

            masked_indices = (
                tokenized_inputs["input_ids"] == self.tokenizer.mask_token_id
            ).bool()

            output.update(
                {
                    "loss": mlm_scores.get("loss", None),
                    "logit": mlm_scores["logits"][masked_indices],
                    "y": tokenized_inputs["labels"][masked_indices],
                }
            )

        return output

    def torch_mask_tokens(
        self, inputs: Any, special_tokens_mask: Optional[Any] = None
    ) -> Tuple[Any, Any]:
        """
        Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L607

        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(
            labels.shape, self.args.mlm_probability, device=self.token_type_ids.device
        )
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(
                torch.full(labels.shape, 0.8, device=self.token_type_ids.device)
            ).bool()
            & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(
                torch.full(labels.shape, 0.5, device=self.token_type_ids.device)
            ).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer),
            labels.shape,
            dtype=torch.long,
            device=self.token_type_ids.device,
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        super(TransformerRanker, TransformerRanker).add_args(parser)
        parser.add_argument(
            "--use_cls_token",
            action="store_true",
            default=False,
            help="use cls token as hidden representation of sequence",
        )
        parser.add_argument(
            "--vocab_path",
            type=str,
            default=None,
            required=True,
            help="path to vocab text file required for tokenizer",
        )
        parser.add_argument(
            "--num_hidden_layers",
            type=int,
            default=6,
            help="number of layers in the transformer",
        )
        parser.add_argument(
            "--max_seq_len",
            type=int,
            default=512,
            help="maximum length allowed for the input sequence",
        )
        parser.add_argument(
            "--hidden_size",
            type=int,
            default=256,
            help="maximum length allowed for the input sequence",
        )
        parser.add_argument(
            "--intermediate_size",
            type=int,
            default=512,
            help="maximum length allowed for the input sequence",
        )
        parser.add_argument(
            "--num_heads",
            type=int,
            default=8,
            help="maximum length allowed for the input sequence",
        )
        parser.add_argument(
            "--do_masked_language_model",
            "-mlm",
            action="store_true",
            default=False,
            help="whether to perform masked language model task",
        )
        parser.add_argument(
            "--mlm_probability",
            type=float,
            default=0.1,
            help="probability that a token chosen to be masked. IF chosen, 80% will be masked, 10% random, 10% original",
        )
        parser.add_argument(
            "--mlm_model_path",
            type=str,
            default=None,
            help="path to saved model if loading from pretrained",
        )


@register_object("wldn_transformer", "model")
class WLDNTransformer(WLDN):
    def __init__(self, args):
        super().__init__(args)  # gives self.wln (GAT)
        self.esm_tokenizer = AutoTokenizer.from_pretrained(args.esm_model_version)
        self.esm_model = EsmModel.from_pretrained(args.esm_model_version)
        self.freeze_encoder = args.freeze_encoder
        if self.freeze_encoder:
            self.esm_model.requires_grad_(False)
            self.esm_model.eval()
        econfig = self.get_transformer_config(args)
        econfig.is_decoder = False
        self.model = BertModel(econfig, add_pooling_layer=True)
        self.config = self.model.config
        self.args = args
        self.register_buffer("token_type_ids", torch.zeros(1, dtype=torch.long))

        self.final_transform = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, 1),
        )  # need to overwrite because output is different size

    def get_transformer_config(self, args):
        bert_config = BertConfig(
            # vocab_size=self.bert_tokenizer.vocab_size,
            max_position_embeddings=args.max_seq_len,
            type_vocab_size=2,
            output_hidden_states=True,
            output_attentions=True,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            num_attention_heads=args.num_heads,
            num_hidden_layers=args.num_hidden_layers,
        )
        return bert_config

    def encode_sequence(self, batch):
        encoder_input_ids = self.esm_tokenizer(
            batch["sequence"],
            padding="longest",
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        for k, v in encoder_input_ids.items():
            encoder_input_ids[k] = v.to(self.token_type_ids.device)

        if self.freeze_encoder:
            self.esm_model.requires_grad_(False)
            with torch.no_grad():
                encoder_outputs = self.esm_model(
                    input_ids=encoder_input_ids["input_ids"],
                    attention_mask=encoder_input_ids["attention_mask"],
                )
        else:
            encoder_outputs = self.esm_model(
                input_ids=encoder_input_ids["input_ids"],
                attention_mask=encoder_input_ids["attention_mask"],
            )
        encoder_hidden_states = encoder_outputs["pooler_output"].unsqueeze(1)
        mask = encoder_input_ids["attention_mask"][:, 0:1]
        return encoder_hidden_states, mask

    def forward(self, batch):
        prot_feats, prot_attn = self.encode_sequence(batch)  # 1 x len_seq x hidden_dim

        product_candidates_list = self.get_product_candidate_list(
            batch, batch["row_id"]
        )

        reactant_node_feats = self.wln(batch["reactants"])[
            "node_features"
        ]  # N x D, where N is all the nodes in the batch
        dense_reactant_node_feats, mask = to_dense_batch(
            reactant_node_feats, batch=batch["reactants"].batch
        )  # B x max_batch_N x D
        candidate_scores = []
        for idx, product_candidates in enumerate(product_candidates_list):
            # get node features for candidate products
            product_candidates = product_candidates.to(reactant_node_feats.device)
            candidate_node_feats = self.wln(product_candidates)["node_features"]
            dense_candidate_node_feats, candidate_mask = to_dense_batch(
                candidate_node_feats, batch=product_candidates.batch
            )  # B x num_nodes x D

            num_nodes = dense_candidate_node_feats.shape[1]

            # compute difference vectors and replace the node features of the product graph with them
            diff_node_feats = dense_candidate_node_feats - dense_reactant_node_feats[
                idx
            ][:num_nodes].unsqueeze(0)

            num_candidates = diff_node_feats.shape[0]
            # repeat protein features and mask to shape
            repeated_prot = prot_feats[idx].unsqueeze(0).repeat(num_candidates, 1, 1)
            prot_mask = prot_attn[idx].unsqueeze(0).repeat(num_candidates, 1)

            # concatenate the prot features with the product features
            concatenated_feats = torch.cat(
                [diff_node_feats, repeated_prot], dim=1
            )  # num_candidates x (max_batch_N + len_seq) x D

            # create token_type_ids tensor
            token_type_ids = self.token_type_ids.repeat(
                *concatenated_feats.shape[:2]
            )  # Initialize with zeros
            token_type_ids[
                :, num_nodes:
            ] = 1  # Assign token type 1 to the second sequence

            # compute the attention mask so that padded product features are not attended to
            attention_mask = torch.cat(
                [candidate_mask, prot_mask], dim=1
            )  # num_candidates x (max_batch_N + len_seq)

            outputs = self.model(
                inputs_embeds=concatenated_feats,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            score = self.final_transform(outputs["pooler_output"])
            if self.args.add_core_score:
                core_scores = [
                    sum(c[-1] for c in cand_changes)
                    for cand_changes in product_candidates.candidate_bond_change
                ]
                score = score + torch.tensor(
                    core_scores, device=score.device
                ).unsqueeze(-1)
            candidate_scores.append(score)  # K x 1

        output = {
            "candidate_logit": candidate_scores,
            "product_candidates_list": product_candidates_list,
        }
        return output

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        super(WLDNTransformer, WLDNTransformer).add_args(parser)
        parser.add_argument(
            "--esm_model_version",
            type=str,
            default=None,
            help="which version of ESM to use",
        )
        parser.add_argument(
            "--freeze_encoder",
            action="store_true",
            default=False,
            help="whether use model as pre-trained encoder and not update weights",
        )
        parser.add_argument(
            "--num_hidden_layers",
            type=int,
            default=6,
            help="number of layers in the transformer",
        )
        parser.add_argument(
            "--max_seq_len",
            type=int,
            default=512,
            help="maximum length allowed for the input sequence",
        )
        parser.add_argument(
            "--hidden_size",
            type=int,
            default=256,
            help="maximum length allowed for the input sequence",
        )
        parser.add_argument(
            "--intermediate_size",
            type=int,
            default=512,
            help="maximum length allowed for the input sequence",
        )
        parser.add_argument(
            "--num_heads",
            type=int,
            default=8,
            help="maximum length allowed for the input sequence",
        )
