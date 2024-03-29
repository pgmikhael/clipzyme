import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from typing import Dict
from torch_geometric.utils import to_dense_batch, to_dense_adj, dense_to_sparse
from torch_scatter import scatter
from esm import pretrained
from clipzyme.utils.classes import set_nox_type
from clipzyme.utils.registry import register_object, get_object
from clipzyme.utils.pyg import x_map
from clipzyme.models.abstract import AbstractModel
from clipzyme.models.chemprop import DMPNNEncoder


@register_object("enzyme_reaction_clip", "model")
class EnzymeReactionCLIP(AbstractModel):
    def __init__(self, args):
        super(EnzymeReactionCLIP, self).__init__()
        self.args = args
        self.reaction_clip_model_path = copy.copy(args.reaction_clip_model_path)
        self.use_as_protein_encoder = getattr(args, "use_as_protein_encoder", False)
        self.use_as_mol_encoder = getattr(args, "use_as_mol_encoder", False) or getattr(
            args, "use_as_reaction_encoder", False
        )  # keep mol for backward compatibility
        args.train_esm_with_graph = getattr(args, "train_esm_with_graph", False)

        if args.reaction_clip_model_path is not None:
            state_dict = torch.load(args.reaction_clip_model_path)
            state_dict_copy = {
                k.replace("model.", "", 1): v
                for k, v in state_dict["state_dict"].items()
            }
            args = state_dict["hyper_parameters"]["args"]

        self.protein_encoder = get_object(args.protein_encoder, "model")(args)
        # option to train esm
        if args.train_esm_with_graph:
            self.esm_dir = args.train_esm_dir
            model, alphabet = pretrained.load_model_and_alphabet(args.train_esm_dir)
            self.esm_model = model
            self.alphabet = alphabet
            self.batch_converter = alphabet.get_batch_converter()

        self.ln_final = nn.LayerNorm(
            args.chemprop_hidden_dim
            if not args.use_protein_graphs
            else args.protein_dim
        )  # needs to be shape of protein_hidden, make it chemprop shape since we typically make these match

        self.logit_scale = nn.Parameter(
            torch.ones([]) * torch.log(torch.tensor(1 / 0.07))
        )

        wln_diff_args = copy.deepcopy(args)
        if args.model_name != "enzyme_reaction_clip_wldnv1":
            self.wln = DMPNNEncoder(args)  # WLN for mol representation
            wln_diff_args = copy.deepcopy(args)
            wln_diff_args.chemprop_edge_dim = args.chemprop_hidden_dim
            # wln_diff_args.chemprop_num_layers = 1
            self.wln_diff = DMPNNEncoder(wln_diff_args)

            # mol: attention pool
            self.final_linear = nn.Linear(
                args.chemprop_hidden_dim, args.chemprop_hidden_dim, bias=False
            )
            self.attention_fc = nn.Linear(args.chemprop_hidden_dim, 1, bias=False)

        # classifier
        if args.do_matching_task:
            self.mlp = get_object(args.mlp_name, "model")(args)

        if self.reaction_clip_model_path is not None:
            self.load_state_dict(state_dict_copy)

        if self.args.clip_freeze_esm:
            self.protein_encoder.requires_grad_(False)

    def encode_protein(self, batch):
        if self.args.use_protein_graphs:
            if self.args.train_esm_with_graph:
                sequences = [
                    (i, s) for i, s in enumerate(batch["graph"].structure_sequence)
                ]
                repr_layer = len(self.esm_model.layers)
                _, _, batch_tokens = self.batch_converter(sequences)
                batch_tokens = batch_tokens.to(self.logit_scale.device)
                mask = torch.logical_and(
                    torch.logical_and(
                        (batch_tokens != self.alphabet.cls_idx),
                        (batch_tokens != self.alphabet.eos_idx),
                    ),
                    (batch_tokens != self.alphabet.padding_idx),
                )
                out = self.esm_model(batch_tokens, repr_layers=[repr_layer])
                representations = out["representations"][repr_layer][mask]
                batch["graph"]["receptor"].x = representations
                if self.args.use_protein_msa:
                    msa = batch["graph"]["receptor"].msa
                    batch["graph"]["receptor"].x = torch.concat(
                        [representations, msa], dim=-1
                    )

            feats, coors = self.protein_encoder(batch)
            try:
                batch_idxs = batch["graph"]["receptor"].batch
            except:
                batch_idxs = batch["receptor"].batch
            protein_features = scatter(
                feats,
                batch_idxs,
                dim=0,
                reduce=self.args.pool_type,
            )
        else:
            if self.args.clip_freeze_esm:
                self.protein_encoder.requires_grad_(False)
                with torch.no_grad():
                    protein_features = self.protein_encoder(
                        {
                            "x": batch["sequence"],
                            "sequence": batch["sequence"],
                            "batch": batch,
                        }
                    )["hidden"]
            else:
                protein_features = self.protein_encoder(
                    {
                        "x": batch["sequence"],
                        "sequence": batch["sequence"],
                        "batch": batch,
                    }
                )["hidden"]

        # apply normalization
        protein_features = self.ln_final(protein_features)
        return protein_features

    def encode_reaction(self, batch):
        reactant_edge_feats = self.wln(batch["reactants"])[
            "edge_features"
        ]  # N x D, where N is all the nodes in the batch
        product_edge_feats = self.wln(batch["products"])[
            "edge_features"
        ]  # N x D, where N is all the nodes in the batch

        dense_reactant_edge_feats = to_dense_adj(
            edge_index=batch["reactants"].edge_index,
            edge_attr=reactant_edge_feats,
            batch=batch["reactants"].batch,
        )
        dense_product_edge_feats = to_dense_adj(
            edge_index=batch["products"].edge_index,
            edge_attr=product_edge_feats,
            batch=batch["products"].batch,
        )
        sum_vectors = dense_reactant_edge_feats + dense_product_edge_feats

        # undensify
        flat_sum_vectors = sum_vectors.sum(-1)
        new_edge_indices = [dense_to_sparse(E)[0] for E in flat_sum_vectors]
        new_edge_attr = torch.vstack(
            [sum_vectors[i, e[0], e[1]] for i, e in enumerate(new_edge_indices)]
        )
        cum_num_nodes = torch.cumsum(torch.bincount(batch["reactants"].batch), 0)
        new_edge_index = torch.hstack(
            [new_edge_indices[0]]
            + [ei + cum_num_nodes[i] for i, ei in enumerate(new_edge_indices[1:])]
        )
        reactants_and_products = batch["reactants"]
        reactants_and_products.edge_attr = new_edge_attr
        reactants_and_products.edge_index = new_edge_index

        # apply a separate WLN to the difference graph
        wln_diff_output = self.wln_diff(reactants_and_products)

        if self.args.aggregate_over_edges:
            edge_feats = wln_diff_output["edge_features"]
            edge_batch = batch["reactants"].batch[new_edge_index[0]]
            graph_feats = scatter(edge_feats, edge_batch, dim=0, reduce="sum")
        else:
            sum_node_feats = wln_diff_output["node_features"]
            sum_node_feats, _ = to_dense_batch(
                sum_node_feats, batch["products"].batch
            )  # num_candidates x max_num_nodes x D
            graph_feats = torch.sum(sum_node_feats, dim=-2)
        return graph_feats

    def forward(self, batch) -> Dict:
        output = {}
        if getattr(self.args, "use_as_protein_encoder", False):
            protein_features = self.encode_protein(batch)

            protein_features = protein_features / protein_features.norm(
                dim=1, keepdim=True
            )

            output.update(
                {
                    "hidden": protein_features,
                }
            )
            return output

        if getattr(self.args, "use_as_mol_encoder", False) or getattr(
            self.args, "use_as_reaction_encoder", False
        ):
            substrate_features = self.encode_reaction(batch)
            substrate_features = substrate_features / substrate_features.norm(
                dim=1, keepdim=True
            )
            output.update(
                {
                    "hidden": substrate_features,
                }
            )
            return output

        substrate_features = self.encode_reaction(batch)
        protein_features = self.encode_protein(batch)

        # normalized features
        substrate_features = substrate_features / substrate_features.norm(
            dim=1, keepdim=True
        )
        protein_features = protein_features / protein_features.norm(dim=1, keepdim=True)

        output.update(
            {
                "substrate_hiddens": substrate_features,
                "protein_hiddens": protein_features,
            }
        )

        if self.args.do_matching_task:
            # take negatives based on EC
            ec = batch["ec2"] != batch["ec1"][:, None]
            ec = (ec / ec.sum(1))[:, None]
            neg_idx = []
            for e in ec:
                neg_idx.append(torch.multinomial(e, 1))
            neg_idx = torch.concat(neg_idx).squeeze()

            # take pairwise similarity of rxn embed and choose negatives
            # substrate_sim = substrate_features @ substrate_features.T
            # neg_idx = torch.argmin(substrate_sim, dim=-1)
            neg_samples = substrate_features[neg_idx]

            concat_hiddens_pos = torch.cat(
                [protein_features, substrate_features], dim=-1
            )
            concat_hiddens_neg = torch.cat([protein_features, neg_samples], dim=-1)
            concat_hiddens = torch.cat([concat_hiddens_pos, concat_hiddens_neg], dim=0)
            output["logit"] = self.mlp({"x": concat_hiddens})["logit"]
            bs = concat_hiddens_pos.shape[0]
            output["y"] = torch.cat(
                [concat_hiddens.new_ones(bs), concat_hiddens.new_zeros(bs)], dim=0
            )

        return output

    @staticmethod
    def add_args(parser) -> None:
        DMPNNEncoder.add_args(parser)
        parser.add_argument(
            "--mlp_name",
            type=str,
            action=set_nox_type("model"),
            default="mlp_classifier",
            help="Name of mlp to use",
        )
        parser.add_argument(
            "--do_matching_task",
            action="store_true",
            default=False,
            help="do molecule-protein matching",
        )
        parser.add_argument(
            "--protein_encoder",
            type=str,
            action=set_nox_type("model"),
            default="fair_esm2",
            help="Name of encoder to use",
        )
        parser.add_argument(
            "--reaction_clip_model_path",
            type=str,
            default=None,
            help="path to saved model if loading from pretrained",
        )
        parser.add_argument(
            "--aggregate_over_edges",
            action="store_true",
            default=False,
            help="use gat implementation.",
        )
        parser.add_argument(
            "--clip_freeze_esm",
            action="store_true",
            default=False,
            help="use gat implementation.",
        )
        parser.add_argument(
            "--use_as_protein_encoder",
            action="store_true",
            default=False,
            help="use gat implementation.",
        )
        parser.add_argument(
            "--use_as_mol_encoder",
            action="store_true",
            default=False,
            help="use gat implementation.",
        )
        parser.add_argument(
            "--train_esm_with_graph",
            action="store_true",
            default=False,
            help="train ESM model with graph NN.",
        )
        parser.add_argument(
            "--train_esm_dir",
            type=str,
            default="/home/snapshots/metabolomics/esm2/checkpoints/esm2_t33_650M_UR50D.pt",
            help="directory to load esm model from",
        )


@register_object("enzyme_reaction_clip_ec", "model")
class EnzymeReactionCLIPEC(EnzymeReactionCLIP):
    def __init__(self, args):
        super(EnzymeReactionCLIPEC, self).__init__(args)
        # the idea being that this protein_encoder produces both the protein and ec predictions
        # if pretrained ec model, load it
        if args.ec_model_model_path is not None:  # load pretrained model
            state_dict_ec = torch.load(args.ec_model_model_path)
            state_dict_ec_copy = {
                k.replace("model.", "", 1): v
                for k, v in state_dict_ec["state_dict"].items()
            }
            ec_args = state_dict_ec["hyper_parameters"]["args"]
            assert (
                args.protein_encoder == ec_args.protein_encoder
            ), "ec model name must match"
            assert (
                args.vocab_path == ec_args.vocab_path
            ), "pretrained model has different vocab"
            self.protein_encoder = get_object(args.protein_encoder, "model")(ec_args)
            self.protein_encoder.load_state_dict(state_dict_ec_copy)

    def encode_protein(self, batch):
        if self.args.use_protein_graphs and self.args.train_esm_with_graph:
            sequences = [
                (i, s) for i, s in enumerate(batch["graph"].structure_sequence)
            ]
            repr_layer = len(self.esm_model.layers)
            _, _, batch_tokens = self.batch_converter(sequences)
            batch_tokens = batch_tokens.to(self.logit_scale.device)
            mask = torch.logical_and(
                torch.logical_and(
                    (batch_tokens != self.alphabet.cls_idx),
                    (batch_tokens != self.alphabet.eos_idx),
                ),
                (batch_tokens != self.alphabet.padding_idx),
            )
            out = self.esm_model(batch_tokens, repr_layers=[repr_layer])
            representations = out["representations"][repr_layer][mask]
            batch["graph"]["receptor"].x = representations
            if self.args.use_protein_msa:
                msa = batch["graph"]["receptor"].msa
                batch["graph"]["receptor"].x = torch.concat(
                    [representations, msa], dim=-1
                )

        protein_output = self.protein_encoder(batch)
        # apply normalization
        protein_output["protein_features"] = self.ln_final(
            protein_output["protein_features"]
        )
        return protein_output

    def forward(self, batch) -> Dict:
        output = {}
        if getattr(self.args, "use_as_protein_encoder", False):
            encoded_protein_output = self.encode_protein(batch)
            protein_features = encoded_protein_output["protein_features"]

            protein_features = protein_features / protein_features.norm(
                dim=1, keepdim=True
            )

            output.update(
                {
                    "hidden": protein_features,
                }
            )
            return output

        if getattr(self.args, "use_as_mol_encoder", False):
            substrate_features = self.encode_reaction(batch)
            substrate_features = substrate_features / substrate_features.norm(
                dim=1, keepdim=True
            )
            output.update(
                {
                    "hidden": substrate_features,
                }
            )
            return output

        substrate_features = self.encode_reaction(batch)
        encoded_protein_output = self.encode_protein(batch)
        protein_features = encoded_protein_output["protein_features"]

        # normalized features
        substrate_features = substrate_features / substrate_features.norm(
            dim=1, keepdim=True
        )
        protein_features = protein_features / protein_features.norm(dim=1, keepdim=True)

        encoded_protein_output["protein_hiddens"] = protein_features
        encoded_protein_output["substrate_hiddens"] = substrate_features

        return encoded_protein_output

    @staticmethod
    def add_args(parser) -> None:
        DMPNNEncoder.add_args(parser)
        parser.add_argument(
            "--mlp_name",
            type=str,
            action=set_nox_type("model"),
            default="mlp_classifier",
            help="Name of mlp to use",
        )
        parser.add_argument(
            "--do_matching_task",
            action="store_true",
            default=False,
            help="do molecule-protein matching",
        )
        parser.add_argument(
            "--protein_encoder",
            type=str,
            action=set_nox_type("model"),
            default="fair_esm2",
            help="Name of encoder to use",
        )
        parser.add_argument(
            "--reaction_clip_model_path",
            type=str,
            default=None,
            help="path to saved model if loading from pretrained",
        )
        parser.add_argument(
            "--aggregate_over_edges",
            action="store_true",
            default=False,
            help="use gat implementation.",
        )
        parser.add_argument(
            "--clip_freeze_esm",
            action="store_true",
            default=False,
            help="use gat implementation.",
        )
        parser.add_argument(
            "--use_as_protein_encoder",
            action="store_true",
            default=False,
            help="use gat implementation.",
        )
        parser.add_argument(
            "--use_as_mol_encoder",
            action="store_true",
            default=False,
            help="use gat implementation.",
        )
        parser.add_argument(
            "--do_ec_task",
            action="store_true",
            default=False,
            help="do ec prediction",
        )
        parser.add_argument(
            "--ec_model_name",
            type=str,
            action=set_nox_type("model"),
            default="classifier",
            help="Name of model to use to predict ECs",
        )
        parser.add_argument(
            "--ec_model_model_path",
            type=str,
            default=None,
            help="path to saved model if loading from pretrained",
        )
        parser.add_argument(
            "--train_esm_with_graph",
            action="store_true",
            default=False,
            help="train ESM model with graph NN.",
        )


@register_object("enzyme_reaction_clip_cgr", "model")
class EnzymeReactionCLIPv2(EnzymeReactionCLIP):
    def encode_reaction(self, batch):
        dense_reactant_edge_feats = to_dense_adj(
            edge_index=batch["reactants"].edge_index,
            edge_attr=batch[
                "reactants"
            ].edge_attr,  # ! add 1 since encoding no edge here REMOVE with one-hot encoding
            batch=batch["reactants"].batch,
        )

        dense_product_edge_feats = to_dense_adj(
            edge_index=batch["products"].edge_index,
            edge_attr=batch[
                "products"
            ].edge_attr,  # ! add 1 since encoding no edge here
            batch=batch["products"].batch,
        )

        # node features
        # cgr_nodes = torch.cat([batch["reactants"].x, batch["products"].x], dim = -1)

        node_diff = batch["reactants"].x - batch["products"].x
        cgr_nodes = torch.cat(
            [batch["reactants"].x, node_diff[:, len(x_map["atomic_num"]) :]], dim=-1
        )

        # edge features
        # cgr_attr = torch.cat([dense_reactant_edge_feats, dense_product_edge_feats], dim = -1) # B, N, N, D
        cgr_attr = torch.cat(
            [
                dense_reactant_edge_feats,
                dense_reactant_edge_feats - dense_product_edge_feats,
            ],
            dim=-1,
        )

        # undensify
        flat_sum_vectors = cgr_attr.sum(-1)
        new_edge_indices = [dense_to_sparse(E)[0] for E in flat_sum_vectors]
        new_edge_attr = torch.vstack(
            [cgr_attr[i, e[0], e[1]] for i, e in enumerate(new_edge_indices)]
        )
        cum_num_nodes = torch.cumsum(torch.bincount(batch["reactants"].batch), 0)
        new_edge_index = torch.hstack(
            [new_edge_indices[0]]
            + [ei + cum_num_nodes[i] for i, ei in enumerate(new_edge_indices[1:])]
        )

        # make graph
        reactants_and_products = batch["reactants"]
        reactants_and_products.x = cgr_nodes
        reactants_and_products.edge_attr = new_edge_attr
        reactants_and_products.edge_index = new_edge_index

        # apply a separate WLN to the difference graph
        wln_diff_output = self.wln(reactants_and_products)

        if self.args.aggregate_over_edges:
            edge_feats = self.final_linear(wln_diff_output["edge_features"])
            edge_batch = batch["reactants"].batch[new_edge_index[0]]
            edge_feats, edge_mask = to_dense_batch(edge_feats, edge_batch)
            attn = self.attention_fc(edge_feats)
            attn[~edge_mask] = -torch.inf
            attn = torch.softmax(attn, -2)
            # graph_feats = scatter(edge_feats, edge_batch, dim = 0, reduce="sum")
            graph_feats = torch.sum(edge_feats * attn, dim=-2)
        else:
            node_feats = self.final_linear(wln_diff_output["node_features"])
            node_feats, node_mask = to_dense_batch(node_feats, batch["reactants"].batch)
            attn = self.attention_fc(node_feats)
            attn[~node_mask] = -torch.inf
            attn = torch.softmax(attn, -2)
            # node_feats, _ = to_dense_batch(node_feats, batch["reactants"].batch) # num_candidates x max_num_nodes x D
            graph_feats = torch.sum(node_feats * attn, dim=-2)
        return graph_feats


@register_object("enzyme_reaction_clip_string", "model")
class EnzymeReactionCLIPString(EnzymeReactionCLIP):
    def __init__(self, args):
        super(EnzymeReactionCLIPString, self).__init__(args)
        self.substrate_encoder = get_object(args.substrate_encoder, "model")(args)

    def encode_reaction(self, batch):
        feats = self.substrate_encoder(batch)["encoder_output"]
        return feats

    @staticmethod
    def add_args(parser) -> None:
        super(EnzymeReactionCLIPString, EnzymeReactionCLIPString).add_args(parser)
        parser.add_argument(
            "--substrate_encoder",
            type=str,
            action=set_nox_type("model"),
            default="full_reaction_encoder",
            help="Rxn String Encoder",
        )


@register_object("enzyme_reaction_clip_pretrained", "model")
class EnzymeReactionCLIPPretrained(EnzymeReactionCLIP):
    def __init__(self, args):
        super(EnzymeReactionCLIPPretrained, self).__init__(args)
        self.substrate_encoder = get_object(args.substrate_encoder, "model")(args)
        if args.substrate_model_path is not None:
            state_dict = torch.load(args.substrate_model_path)
            state_dict_copy = {
                k.replace("model.", "", 1): v
                for k, v in state_dict["state_dict"].items()
            }
            # Remove keys from state_dict that are not in the model
            model_state_dict_keys = set(self.substrate_encoder.state_dict().keys())
            state_dict_keys = list(
                state_dict_copy.keys()
            )  # We use list to avoid RuntimeError for changing dict size during iteration
            for key in state_dict_keys:
                if key not in model_state_dict_keys:
                    del state_dict_copy[key]
            self.substrate_encoder.load_state_dict(state_dict_copy)

        if args.chemprop_hidden_dim != args.protein_dim:
            self.substrate_projection = nn.Linear(
                args.chemprop_hidden_dim, args.protein_dim
            )  # needs to be shape of protein_hidden, make it chemprop shape since we typically make these match

    def encode_reaction(self, batch):
        feats = self.substrate_encoder(batch)
        # unbatch the graph
        feats, mask = to_dense_batch(feats["c_final"], batch=batch["mol"].batch)
        feats = feats.sum(1)  # sum over all nodes
        feats = self.substrate_projection(feats)
        return feats

    @staticmethod
    def add_args(parser) -> None:
        super(EnzymeReactionCLIPPretrained, EnzymeReactionCLIPPretrained).add_args(
            parser
        )
        parser.add_argument(
            "--substrate_encoder",
            type=str,
            action=set_nox_type("model"),
            default="full_reaction_encoder",
            help="Rxn String Encoder",
        )
        parser.add_argument(
            "--substrate_model_path",
            type=str,
            default=None,
            help="path to saved model if loading from pretrained",
        )


@register_object("enzyme_reaction_clip_wldn", "model")
class EnzymeReactionCLIPWLDN(EnzymeReactionCLIPPretrained):
    def __init__(self, args):
        super(EnzymeReactionCLIPWLDN, self).__init__(args)
        del self.substrate_encoder.reactivity_net

    def encode_reaction(self, batch):
        gat_output = self.substrate_encoder.wln(batch["reactants"])
        reactant_node_feats = gat_output["node_features"]

        gat_output = self.substrate_encoder.wln(batch["products"])
        product_node_feats = gat_output["node_features"]

        difference_vectors = product_node_feats - reactant_node_feats

        product_graph = batch["products"].clone()
        product_graph.x = difference_vectors

        # apply a separate WLN to the difference graph
        wln_diff_output = self.substrate_encoder.wln_diff(product_graph)
        diff_node_feats = wln_diff_output["node_features"]
        diff_node_feats, mask = to_dense_batch(
            diff_node_feats, batch=product_graph.batch
        )
        feats = diff_node_feats.sum(1)  # sum over all nodes
        if feats.shape[-1] != self.args.protein_dim:
            feats = self.substrate_projection(feats)
        return feats

    def add_args(parser) -> None:
        parser.add_argument(
            "--substrate_model_path",
            type=str,
            default=None,
            help="path to saved model if loading from pretrained",
        )
        parser.add_argument(
            "--substrate_encoder",
            type=str,
            action=set_nox_type("model"),
            default="full_reaction_encoder",
            help="Rxn String Encoder",
        )
        parser.add_argument(
            "--mlp_name",
            type=str,
            action=set_nox_type("model"),
            default="mlp_classifier",
            help="Name of mlp to use",
        )
        parser.add_argument(
            "--do_matching_task",
            action="store_true",
            default=False,
            help="do molecule-protein matching",
        )
        parser.add_argument(
            "--protein_encoder",
            type=str,
            action=set_nox_type("model"),
            default="fair_esm2",
            help="Name of encoder to use",
        )
        parser.add_argument(
            "--reaction_clip_model_path",
            type=str,
            default=None,
            help="path to saved model if loading from pretrained",
        )
        parser.add_argument(
            "--aggregate_over_edges",
            action="store_true",
            default=False,
            help="use gat implementation.",
        )
        parser.add_argument(
            "--clip_freeze_esm",
            action="store_true",
            default=False,
            help="use gat implementation.",
        )
        parser.add_argument(
            "--use_as_protein_encoder",
            action="store_true",
            default=False,
            help="use gat implementation.",
        )
        parser.add_argument(
            "--use_as_mol_encoder",
            action="store_true",
            default=False,
            help="use gat implementation.",
        )
        parser.add_argument(
            "--train_esm_with_graph",
            action="store_true",
            default=False,
            help="train ESM model with graph NN.",
        )
        parser.add_argument(
            "--train_esm_dir",
            type=str,
            default="/home/snapshots/metabolomics/esm2/checkpoints/esm2_t33_650M_UR50D.pt",
            help="directory to load esm model from",
        )
