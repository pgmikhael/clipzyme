import torch
import torch.nn as nn
import torch.nn.functional as F
from clipzyme.utils.registry import get_object, register_object
from clipzyme.utils.classes import set_nox_type
from clipzyme.utils.wln_processing import (
    generate_candidates_from_scores,
    get_batch_candidate_bonds,
    robust_edit_mol,
)
from clipzyme.models.abstract import AbstractModel
from torch_scatter import scatter_add
from torch_geometric.utils import to_dense_batch, to_dense_adj
from clipzyme.models.gat import GAT
from clipzyme.models.chemprop import WLNEncoder, DMPNNEncoder
from rdkit import Chem
import copy
import os
from rich import print as rprint


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
        GAT.add_args(parser)
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
        AbstractModel.add_args(parser)
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
            rprint(
                f"[magenta]Loaded Reactivity Model from {args.reactivity_model_path}[/magenta]"
            )
        except:
            self.reactivity_net = get_object(args.reactivity_net_type, "model")(
                args
            ).requires_grad_(False)
            rprint("[magenta]WARNING: Could not load pretrained model[/magenta]")
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
            if args.model_name == "wldn":
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

        if getattr(args, "load_wln_cache_in_dataset", False):
            self.use_cache = (args.cache_path is not None) and (
                not args.load_wln_cache_in_dataset
            )
            if self.use_cache:
                self.cache = WLDN_Cache(os.path.join(args.cache_path), "pt")
        if self.args.test:
            assert not self.args.train

        if args.ranker_model_path:
            try:
                state_dict = torch.load(args.ranker_model_path)
                self.wln.load_state_dict(
                    {
                        k[len("model.wln.") :]: v
                        for k, v in state_dict["state_dict"].items()
                        if k.startswith("model.wln.")
                    }
                )
                self.wln_diff.load_state_dict(
                    {
                        k[len("model.wln_diff.") :]: v
                        for k, v in state_dict["state_dict"].items()
                        if k.startswith("model.wln_diff.")
                    }
                )
                self.final_transform.load_state_dict(
                    {
                        k[len("model.final_transform.") :]: v
                        for k, v in state_dict["state_dict"].items()
                        if k.startswith("model.final_transform.")
                    }
                )
                rprint(
                    f"[magenta]Loaded Ranker from {args.ranker_model_path}[/magenta]"
                )
            except:
                pass

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
        AbstractModel.add_args(parser)
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
        parser.add_argument(
            "--ranker_model_path",
            type=str,
            help="path to pretrained ranker model if loading each model separately",
        )
