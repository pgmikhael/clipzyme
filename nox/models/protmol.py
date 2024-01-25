import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from typing import Union, Tuple, Any, List, Dict, Optional
from torch_geometric.utils import to_dense_batch, to_dense_adj, dense_to_sparse
from torch_scatter import scatter
import torch.distributed as dist
from transformers import AutoTokenizer, EsmConfig, BertModel
from transformers.models.esm.modeling_esm import EsmLMHead
from esm import pretrained
from nox.utils.classes import set_nox_type
from nox.utils.registry import register_object, get_object
from nox.utils.pyg import x_map
from nox.utils.loading import concat_all_gather, all_gather_with_grad
from nox.models.abstract import AbstractModel
from nox.models.chemprop import DMPNNEncoder
from nox.models.modeling_esm import EsmModel, EsmLayer


def invert_attention_mask(encoder_attention_mask, dtype=torch.float32):
    """
    Invert an attention mask (e.g., switches 0. and 1.).

    Args:
        encoder_attention_mask (`torch.Tensor`): An attention mask.

    Returns:
        `torch.Tensor`: The inverted attention mask.
    """
    if encoder_attention_mask.dim() == 3:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
    if encoder_attention_mask.dim() == 2:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
    # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
    # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
    # /transformer/transformer_layers.py#L270
    # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
    # encoder_extended_attention_mask.transpose(-1, -2))
    encoder_extended_attention_mask = encoder_extended_attention_mask.to(
        dtype=dtype
    )  # fp16 compatibility
    encoder_extended_attention_mask = (
        1.0 - encoder_extended_attention_mask
    ) * torch.finfo(dtype).min

    return encoder_extended_attention_mask


@register_object("protmol_clip", "model")
class ProteinMoleculeCLIP(AbstractModel):
    def __init__(self, args):
        super(ProteinMoleculeCLIP, self).__init__()
        self.args = args
        self.substrate_encoder = get_object(args.substrate_encoder, "model")(args)
        self.protein_encoder = get_object(args.protein_encoder, "model")(args)
        self.ln_final = nn.LayerNorm(
            args.chemprop_hidden_dim
        )  # needs to be shape of protein_hidden, make it chemprop shape since we typically make these match
        self.logit_scale = nn.Parameter(
            torch.ones([]) * torch.log(torch.tensor(1 / 0.07))
        )
        if args.protmol_clip_model_path is not None:
            state_dict = torch.load(args.protmol_clip_model_path)
            state_dict_copy = {
                k.replace("model.", "", 1): v
                for k, v in state_dict["state_dict"].items()
            }
            self.load_state_dict(state_dict_copy)

    def forward(self, batch) -> Dict:
        output = {}
        substrate_features_out = self.substrate_encoder(batch)
        substrate_features = substrate_features_out["hidden"]

        protein_features = self.protein_encoder(
            {"x": batch.sequence, "sequence": batch.sequence, "batch": batch}
        )["hidden"]
        # apply normalization
        protein_features = self.ln_final(protein_features)

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
        return output

    @staticmethod
    def add_args(parser) -> None:
        parser.add_argument(
            "--substrate_encoder",
            type=str,
            action=set_nox_type("model"),
            default="chemprop",
            help="Name of encoder to use",
        )
        parser.add_argument(
            "--protein_encoder",
            type=str,
            action=set_nox_type("model"),
            default="fair_esm2",
            help="Name of encoder to use",
        )
        parser.add_argument(
            "--protmol_clip_model_path",
            type=str,
            default=None,
            help="path to saved model if loading from pretrained",
        )


@register_object("protmol_classifier", "model")
class ProtMolClassifier(AbstractModel):
    def __init__(self, args):
        super(ProtMolClassifier, self).__init__()
        self.args = args
        self.enzyme_encoder = get_object(args.enzyme_encoder_name, "model")(args)
        self.substrate_encoder = get_object(args.substrate_encoder_name, "model")(args)
        self.mlp = get_object(args.mlp_name, "model")(args)

    def forward(self, batch):
        # encode molecule
        if self.args.freeze_substrate_encoder:
            with torch.no_grad():
                substrate_dict = self.substrate_encoder(batch)
        else:
            substrate_dict = self.substrate_encoder(batch)
        # encode protein -> must have sequence attribute or key
        x = self.convert_batch_to_seq_list(batch)
        if self.args.freeze_enzyme_encoder:
            with torch.no_grad():
                enzyme_dict = self.enzyme_encoder(x)
        else:
            enzyme_dict = self.enzyme_encoder(x)

        hidden = torch.cat((enzyme_dict["hidden"], substrate_dict["hidden"]), dim=1)
        mlp_dict = self.mlp({"x": hidden})
        output = {
            "sequence_hidden": enzyme_dict["hidden"],
            "substrate_hidden": substrate_dict["hidden"],
        }
        output.update(mlp_dict)
        return output

    def convert_batch_to_seq_list(self, batch):
        if hasattr(batch, "sequence"):
            x = batch.sequence
        elif "sequence" in batch:
            x = batch["sequence"]
        else:
            raise ValueError("Batch must have sequence attribute or key")
        return x

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--substrate_encoder_name",
            type=str,
            action=set_nox_type("model"),
            default="chemprop",
            help="Name of molecular encoder to use",
        )
        parser.add_argument(
            "--enzyme_encoder_name",
            type=str,
            action=set_nox_type("model"),
            default="non_canon_net",
            help="Name of enzyme encoder to use",
        )
        parser.add_argument(
            "--mlp_name",
            type=str,
            action=set_nox_type("model"),
            default="mlp_classifier",
            help="Name of mlp to use",
        )
        parser.add_argument(
            "--freeze_substrate_encoder",
            action="store_true",
            default=False,
            help="",
        )
        parser.add_argument(
            "--freeze_enzyme_encoder",
            action="store_true",
            default=False,
            help="",
        )


@register_object("protmol_clip_classifier", "model")
class ProtMolCLIPClassifier(ProtMolClassifier):
    def __init__(self, args):
        super(ProtMolCLIPClassifier, self).__init__(args)
        # load clip model
        self.clip = get_object(args.enzyme_encoder_name, "model")(args)
        self.enzyme_encoder = self.clip.protein_encoder
        self.substrate_encoder = self.clip.substrate_encoder


@register_object("enzyme_reaction_clip", "model")
class EnzymeReactionCLIP(AbstractModel):
    def __init__(self, args):
        super(EnzymeReactionCLIP, self).__init__()
        self.args = args
        self.reaction_clip_model_path = copy.copy(args.reaction_clip_model_path)
        self.use_as_protein_encoder = getattr(args, "use_as_protein_encoder", False)
        self.use_as_mol_encoder = getattr(args, "use_as_mol_encoder", False)
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

        if args.do_ec_task:
            self.logit_ec_scale = nn.Parameter(
                torch.ones([]) * torch.log(torch.tensor(1 / 0.07))
            )
            if args.ec_model_model_path is not None:  # load pretrained model
                state_dict_ec = torch.load(args.ec_model_model_path)
                state_dict_ec_copy = {
                    k.replace("model.", "", 1): v
                    for k, v in state_dict_ec["state_dict"].items()
                }
                ec_args = state_dict_ec["hyper_parameters"]["args"]
                assert (
                    args.ec_model_name == ec_args.ec_model_name
                ), "ec model name must match"
                self.ec_head = get_object(args.ec_model_name, "model")(ec_args)
                self.ec_head.load_state_dict(
                    {
                        k[len("model.") :]: v
                        for k, v in state_dict["state_dict"].items()
                        if k.startswith("model")
                    }
                )
            else:
                self.ec_head = get_object(args.ec_model_name, "model")(args)

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

        if self.args.do_ec_task:
            assert (
                "protein_features" in output
            ), "output must have protein features to predict EC"
            output.update(self.ec_head(output["protein_features"]))

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
        parser.add_argument(
            "--train_esm_dir",
            type=str,
            default="/Mounts/rbg-storage1/snapshots/metabolomics/esm2/checkpoints/esm2_t33_650M_UR50D.pt",
            help="directory to load esm model from",
        )


@register_object("enzyme_reaction_clipv2", "model")
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
        wln_diff_args = copy.deepcopy(args)
        wln_diff_args.chemprop_node_dim = args.chemprop_hidden_dim
        # wln_diff_args.chemprop_num_layers = 1
        self.wln_diff = DMPNNEncoder(wln_diff_args)
        del self.wln

    def encode_reaction(self, batch):
        gat_output = self.substrate_encoder.gat_global_attention(batch["reactants"])
        cs = gat_output["node_features"]
        c_tildes = gat_output["node_features_attn"]  # node contexts
        reactant_node_feats = self.substrate_encoder.lin(
            torch.cat([cs, c_tildes], dim=-1)
        )

        gat_output = self.substrate_encoder.gat_global_attention(batch["products"])
        cs = gat_output["node_features"]
        c_tildes = gat_output["node_features_attn"]  # node contexts
        product_node_feats = self.substrate_encoder.lin(
            torch.cat([cs, c_tildes], dim=-1)
        )

        difference_vectors = product_node_feats - reactant_node_feats

        product_graph = batch["products"].clone()
        product_graph.x = difference_vectors

        # apply a separate WLN to the difference graph
        wln_diff_output = self.wln_diff(product_graph)
        diff_node_feats = wln_diff_output["node_features"]
        diff_node_feats, mask = to_dense_batch(
            diff_node_feats, batch=product_graph.batch
        )
        feats = diff_node_feats.sum(1)  # sum over all nodes
        if feats.shape[-1] != self.args.protein_dim:
            feats = self.substrate_projection(feats)
        return feats


@register_object("protmol_clip_multiobjective_small_cgr_heid", "model")
class ProteinMoleculeCLIPMultiObjSmallCGRHeid(AbstractModel):
    def __init__(self, args):
        super(ProteinMoleculeCLIPMultiObjSmallCGRHeid, self).__init__()
        self.args = args
        self.mlm_probability = args.mlm_probability
        self.use_as_protein_encoder = getattr(args, "use_as_protein_encoder", False)
        self.use_as_mol_encoder = getattr(args, "use_as_mol_encoder", False)
        self.use_as_matching_classifier = getattr(
            args, "use_as_matching_classifier", False
        )

        # protein encoder
        self.protein_tokenizer = AutoTokenizer.from_pretrained(args.esm_model_version)
        config = EsmConfig.from_pretrained(args.esm_model_version)
        config.cross_attention_freq = args.cross_attention_frequency
        if args.use_protein_graphs:
            self.egnn = get_object(args.egnn_name, "model")(args)
            self.egnn_mask = nn.Parameter(torch.randn(1, config.hidden_size))
        else:
            self.protein_encoder = EsmModel.from_pretrained(
                args.esm_model_version, config=config
            )
        args.vocab_size = config.vocab_size
        self.register_buffer("devicevar", torch.zeros(1, dtype=torch.int8))

        # molecule encoder
        self.substrate_encoder = get_object(args.substrate_encoder, "model")(args)
        hidden_size = config.hidden_size

        self.ln_final = nn.LayerNorm(hidden_size)
        self.logit_scale = nn.Parameter(
            torch.ones([]) * torch.log(torch.tensor(1 / 0.07))
        )

        # mol: attention pool
        self.final_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attention_fc = nn.Linear(hidden_size, 1, bias=False)

        # reaction center prediction
        self.reaction_center_head = nn.Linear(2 * hidden_size, 2)
        self.mlm_head = EsmLMHead(config)

        # matching layer
        ca_config = copy.deepcopy(config)
        ca_config.is_decoder = True
        ca_config.add_cross_attention = True
        ca_config.token_dropout = False
        self.matching_layer = EsmLayer(ca_config, 0)

        if args.use_rdkit_features:
            self.matching_pair_head = nn.Sequential(
                nn.Linear(hidden_size + args.rdkit_features_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 2),
            )
        else:
            self.matching_pair_head = nn.Linear(hidden_size, 2)

        self.predict_ecs = getattr(args, "predict_ecs", False)
        if self.predict_ecs:
            ec_args = copy.deepcopy(args)
            ec_args.num_classes = len(args.ec_levels["4"].values())
            self.ec_classifier = get_object(args.ec_classifier, "model")(ec_args)

        if args.protmol_clip_model_path is not None:
            state_dict = torch.load(args.protmol_clip_model_path)
            state_dict_copy = {
                k.replace("model.", "", 1): v
                for k, v in state_dict["state_dict"].items()
            }
            self.load_state_dict(state_dict_copy)

    def encode_reaction_latent(self, batch):
        reactant_edge_feats = self.substrate_encoder(batch["reactants"])[
            "edge_features"
        ]  # N x D, where N is all the nodes in the batch
        product_edge_feats = self.substrate_encoder(batch["products"])[
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
        wln_diff_output = self.cgr_encoder(reactants_and_products)

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
        wln_diff_output["hidden"] = graph_feats
        return wln_diff_output

    def encode_reaction(self, batch):
        dense_reactant_edge_feats = to_dense_adj(
            edge_index=batch["reactants"].edge_index,
            edge_attr=batch["reactants"].edge_attr,
            batch=batch["reactants"].batch,
        )

        dense_product_edge_feats = to_dense_adj(
            edge_index=batch["products"].edge_index,
            edge_attr=batch["products"].edge_attr,
            batch=batch["products"].batch,
        )

        # node features
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
        wln_diff_output = self.substrate_encoder(reactants_and_products)

        output = {}

        node_feats = self.final_linear(wln_diff_output["node_features"])
        output["node_features"] = node_feats
        node_feats, node_mask = to_dense_batch(node_feats, batch["reactants"].batch)
        attn = self.attention_fc(node_feats)
        attn[~node_mask] = -torch.inf
        attn = torch.softmax(attn, -2)
        # node_feats, _ = to_dense_batch(node_feats, batch["reactants"].batch) # num_candidates x max_num_nodes x D
        graph_feats = torch.sum(node_feats * attn, dim=-2)
        output["hidden"] = graph_feats

        return output

    def encode_protein(self, batch) -> Dict:
        output = {}

        encoder_input_ids = self.protein_tokenizer(
            batch["sequence"],
            padding="longest"
            if self.args.use_protein_graphs
            else "max_length",  # change to "max_length"
            return_tensors="pt",
            return_special_tokens_mask=True,
            max_length=self.args.max_protein_length,
            truncation=True,
            add_special_tokens=not self.args.use_protein_graphs,  # do not use cls or eos with graph
        )

        # move to device
        for k, v in encoder_input_ids.items():
            encoder_input_ids[k] = v.to(self.devicevar.device)

        encoder_input_ids["return_dict"] = True

        encoder_input_ids_clone = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in encoder_input_ids.items()
        }  # clone to use in MLM, otherwise it will be used in a forward method then modified which raises grad errors

        if self.args.use_protein_graphs:
            feats, coors = self.egnn(batch)
            try:
                batch_idxs = batch["graph"]["receptor"].batch
            except:
                batch_idxs = batch["receptor"].batch
            encoder_hidden_states, mask = to_dense_batch(feats, batch_idxs)
            hidden = scatter(
                feats,
                batch_idxs,
                dim=0,
                reduce=self.args.pool_type,
            )
            output = {
                "last_hidden_state": encoder_hidden_states,
                "attention_mask": mask,
                "hidden": hidden,
            }
        else:
            special_tokens_mask = encoder_input_ids.pop("special_tokens_mask")
            output = self.protein_encoder(**encoder_input_ids)
            encoder_input_ids["special_tokens_mask"] = special_tokens_mask
            output["hidden"] = output[:, 0]  # cls token
            output["attention_mask"] = encoder_input_ids.attention_mask

        return encoder_input_ids, encoder_input_ids_clone, output

    def set_decoder_status(self, status: bool):
        if not self.args.use_protein_graphs:
            self.protein_encoder.config.is_decoder = status
            self.protein_encoder.is_decoder = status

    def forward(self, batch) -> Dict:
        output = {}

        batch_size = len(batch["sequence"])

        if self.use_as_protein_encoder:
            self.set_decoder_status(False)

            protein_features = self.encode_protein(batch)["hidden"]
            # apply normalization
            protein_features = self.ln_final(protein_features)

            protein_features = protein_features / protein_features.norm(
                dim=1, keepdim=True
            )

            output.update(
                {
                    "hidden": protein_features,
                }
            )
            return output

        if self.use_as_mol_encoder:
            substrate_features = self.encode_reaction(batch)["hidden"]
            substrate_features = substrate_features / substrate_features.norm(
                dim=1, keepdim=True
            )
            output.update(
                {
                    "hidden": substrate_features,
                }
            )
            return output

        if self.use_as_matching_classifier:
            substrate_features = self.substrate_encoder(batch["mol"])
            substrate_node_features = substrate_features["node_features"]
            substrate_graph_features = substrate_features["hidden"]
            substrate_graph_features = (
                substrate_graph_features
                / substrate_graph_features.norm(dim=1, keepdim=True)
            )

            (
                protein_input_dict,
                protein_input_dict_clone,
                protein_features,
            ) = self.encode_protein(batch)
            protein_aa_features = self.ln_final(protein_features["last_hidden_state"])
            protein_aa_features = protein_aa_features / protein_aa_features.norm(
                dim=1, keepdim=True
            )
            protein_aa_attention_mask = protein_features["attention_mask"]

            node_features_dense, node_attention_mask = to_dense_batch(
                substrate_node_features, batch=batch["mol"].batch
            )

            matching_output = self.matching_layer(
                hidden_states=protein_aa_features,
                attention_mask=self.protein_encoder.invert_attention_mask(
                    protein_aa_attention_mask
                ),
                encoder_hidden_states=node_features_dense,
                encoder_attention_mask=self.protein_encoder.invert_attention_mask(
                    node_attention_mask
                ),
            )
            if self.args.use_rdkit_features:
                feats = torch.cat(
                    [
                        matching_output[0],
                        batch["rdkit_features"]
                        .repeat(1, matching_output[0].shape[1], 1)
                        .float(),
                    ],
                    dim=-1,
                )
                matching_logits = self.matching_pair_head(feats)
            else:
                matching_logits = self.matching_pair_head(matching_output[0])
            logit = matching_logits.mean(dim=1)  # average across all
            output.update(
                {
                    "logit": logit,
                    "substrate_hiddens": substrate_graph_features,
                    "protein_hiddens": protein_aa_features,
                }
            )

            return output

        ###============== Contrastive ===================###
        self.set_decoder_status(False)

        substrate_features = self.encode_reaction(batch)
        substrate_node_features = substrate_features["node_features"]
        substrate_graph_features = substrate_features["hidden"]
        substrate_graph_features = (
            substrate_graph_features
            / substrate_graph_features.norm(dim=1, keepdim=True)
        )

        (
            protein_input_dict,
            protein_input_dict_clone,
            protein_features,
        ) = self.encode_protein(batch)
        protein_aa_features = self.ln_final(protein_features["last_hidden_state"])
        protein_aa_features = protein_aa_features / protein_aa_features.norm(
            dim=1, keepdim=True
        )
        protein_aa_attention_mask = protein_input_dict["attention_mask"]
        protein_pooled_features = protein_features["hidden"]

        output.update(
            {
                "substrate_hiddens": substrate_graph_features,
                "protein_hiddens": protein_aa_features,
            }
        )

        ###============== Reaction Nodes ===================###
        # prob( node involved in reaction | protein )
        if self.args.do_reaction_node_task:
            protein_cls = torch.repeat_interleave(
                protein_pooled_features, torch.bincount(batch["mol"].batch), dim=0
            )  # use CLS for protein encoding and repeat for each molecule
            substrate_nodes = torch.cat(
                [substrate_features["node_features"], protein_cls], dim=-1
            )
            reaction_center_logit = self.reaction_center_head(substrate_nodes)
            output["reaction_center_logit"] = reaction_center_logit
            output["reaction_center_labels"] = batch["mol"].reaction_nodes

        ###========== Molecule - Protein Matching ==========###
        if self.args.do_matching_task:
            if (self.args.gather_representations_for_matching) and (
                int(self.args.gpus) > 1
            ):  # gather
                substrate_features_all = concat_all_gather(substrate_graph_features)
                protein_features_all = concat_all_gather(protein_aa_features)
                # protein_input_ids = concat_all_gather(protein_input_dict.input_ids)
                protein_attention_mask = concat_all_gather(
                    protein_features["attention_mask"]
                )

                num_nodes = torch.bincount(batch["mol"].batch)
                num_nodes = concat_all_gather(num_nodes)
                num_nodes = max(num_nodes)
                node_features_dense, node_attention_mask = to_dense_batch(
                    substrate_node_features,
                    batch=batch["mol"].batch,
                    max_num_nodes=num_nodes,
                )
                substrate_node_features_all = all_gather_with_grad(node_features_dense)
                node_attention_mask_all = concat_all_gather(node_attention_mask)

                rank = dist.get_rank()

            else:
                substrate_features_all = substrate_graph_features
                protein_features_all = protein_aa_features

                # protein_input_ids = protein_input_dict.input_ids
                protein_attention_mask = protein_features["attention_mask"]

                node_features_dense, node_attention_mask = to_dense_batch(
                    substrate_node_features, batch=batch["mol"].batch
                )

                substrate_node_features_all = node_features_dense
                node_attention_mask_all = node_attention_mask

                rank = 0

            logits_per_substrate = torch.matmul(
                substrate_graph_features.unsqueeze(1).unsqueeze(1),
                protein_features_all.permute(0, 2, 1),
            ).squeeze()  # num graphs, num proteins, sequence length
            logits_per_substrate, _ = logits_per_substrate.max(-1)

            logits_per_protein = torch.matmul(
                protein_aa_features.unsqueeze(1), substrate_features_all.unsqueeze(-1)
            ).squeeze()  # num proteins, num graphs, sequence length
            logits_per_protein, _ = logits_per_protein.max(-1)

            if len(logits_per_protein.shape) == 1:
                logits_per_protein = logits_per_protein.unsqueeze(0)
            if len(logits_per_substrate.shape) == 1:
                logits_per_substrate = logits_per_substrate.unsqueeze(0)

            with torch.no_grad():
                logits_per_substrate[
                    :, rank * batch_size : rank * batch_size + batch_size
                ].fill_diagonal_(-10000)
                logits_per_protein[
                    :, rank * batch_size : rank * batch_size + batch_size
                ].fill_diagonal_(-10000)

            weights_mol2prot = F.softmax(logits_per_substrate, dim=1)
            weights_prot2mol = F.softmax(logits_per_protein, dim=1)

            # select a negative protein for each molecule
            prot_ids_negatives = []
            prot_atts_negatives = []
            for b in range(batch_size):
                neg_idx = torch.multinomial(weights_mol2prot[b], 1).item()
                prot_ids_negatives.append(protein_features_all[neg_idx])
                prot_atts_negatives.append(protein_attention_mask[neg_idx])
            prot_ids_negatives = torch.stack(prot_ids_negatives, dim=0)
            prot_atts_negatives = torch.stack(prot_atts_negatives, dim=0)

            # select a negative molecule for each protein
            molecule_negatives = []
            molecule_atts_negatives = []
            for b in range(batch_size):
                neg_idx = torch.multinomial(weights_prot2mol[b], 1).item()
                molecule_negatives.append(substrate_node_features_all[neg_idx])
                molecule_atts_negatives.append(node_attention_mask_all[neg_idx])
            molecule_negatives = torch.stack(molecule_negatives, dim=0)
            molecule_atts_negatives = torch.stack(molecule_atts_negatives, dim=0)

            prot_ids_all = torch.cat(
                [protein_aa_features, protein_aa_features, prot_ids_negatives], dim=0
            )  # pos, pos, neg
            prot_atts_all = torch.cat(
                [
                    protein_features["attention_mask"],
                    protein_features["attention_mask"],
                    prot_atts_negatives,
                ],
                dim=0,
            )

            mol_nodes_all = torch.cat(
                [node_features_dense, molecule_negatives, node_features_dense], dim=0
            )  # pos, neg, pos
            mol_atts_all = torch.cat(
                [node_attention_mask, molecule_atts_negatives, node_attention_mask],
                dim=0,
            )

            matching_output = self.matching_layer(
                hidden_states=prot_ids_all,
                attention_mask=invert_attention_mask(prot_atts_all),
                encoder_hidden_states=mol_nodes_all,
                encoder_attention_mask=invert_attention_mask(mol_atts_all),
            )

            matching_logits = self.matching_pair_head(matching_output[0])
            output["matching_logits"] = matching_logits.mean(
                dim=1
            )  # average across all
            output["matching_labels"] = torch.cat(
                [
                    torch.ones(batch_size, dtype=torch.long),
                    torch.zeros(2 * batch_size, dtype=torch.long),
                ],
                dim=0,
            ).to(matching_logits.device)

        ###====================== MLM ======================###
        if self.args.do_mlm_task:
            # prob( masked sequence | reactant ) bidirectional

            node_features_dense, node_attention_mask = to_dense_batch(
                substrate_node_features, batch=batch["mol"].batch
            )  # B x max_batch_N x D

            # use pesto as probs for sampling masked tokens
            sequence_annotation = torch.zeros_like(
                protein_input_dict_clone["input_ids"]
            ).float()
            if self.args.use_protein_graphs:
                annot_begin, annot_end = 0, batch["sequence_annotation"].shape[-1]
            else:
                annot_begin, annot_end = 1, batch["sequence_annotation"].shape[-1] + 1
            sequence_annotation[:, annot_begin:annot_end] = batch["sequence_annotation"]

            # get masked inputs
            masked_inputs, mlm_labels, mlm_attention_mask = self.torch_mask_tokens(
                protein_input_dict_clone["input_ids"],
                protein_input_dict_clone["special_tokens_mask"],
                sequence_annotation,
            )
            if self.args.use_protein_graphs:
                mask_indices = masked_inputs.eq(self.protein_tokenizer.mask_token_id)
                mask_indices = mask_indices[
                    protein_input_dict_clone["attention_mask"].bool()
                ]
                batch["graph"]["receptor"].x[mask_indices] = self.egnn_mask
                feats, coors = self.egnn(batch)
                batch_idxs = batch["graph"]["receptor"].batch
                mlm_output, mlm_attention_mask = to_dense_batch(feats, batch_idxs)
            else:
                mlm_output = self.protein_encoder(
                    input_ids=masked_inputs,
                    attention_mask=mlm_attention_mask,
                    # encoder_hidden_states=node_features_dense,
                    # encoder_attention_mask=node_attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                mlm_output = mlm_output[0]

            matching_output = self.matching_layer(
                hidden_states=mlm_output,
                attention_mask=invert_attention_mask(mlm_attention_mask),
                encoder_hidden_states=node_features_dense,
                encoder_attention_mask=invert_attention_mask(node_attention_mask),
            )
            sequence_output = matching_output[0]
            prediction_scores = self.mlm_head(sequence_output)
            output[
                "mlm_logits"
            ] = prediction_scores  # .view(-1, self.config.vocab_size)
            output["mlm_labels"] = mlm_labels  # .view(-1)

        ###====================== EC Prediction ======================###
        if self.predict_ecs:
            output["ec_logits"] = self.ec_classifier({"x": protein_pooled_features})[
                "logit"
            ]

        return output

    def torch_mask_tokens(self, inputs, special_tokens_mask, probability_matrix=None):
        """
        Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L607

        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        if probability_matrix is None:
            probability_matrix = torch.full(
                labels.shape, self.mlm_probability, device=self.devicevar.device
            )

        if special_tokens_mask is None:
            special_tokens_mask = [
                self.protein_tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        if self.protein_tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.protein_tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()

        attention_mask = (~masked_indices).float()
        if self.protein_tokenizer._pad_token is not None:
            attention_padding_mask = labels.eq(self.protein_tokenizer.pad_token_id)
            attention_mask.masked_fill_(attention_padding_mask, value=1.0)

        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(
                torch.full(labels.shape, 0.8, device=self.devicevar.device)
            ).bool()
            & masked_indices
        )

        inputs[indices_replaced] = self.protein_tokenizer.convert_tokens_to_ids(
            self.protein_tokenizer.mask_token
        )

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(
                torch.full(labels.shape, 0.5, device=self.devicevar.device)
            ).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.protein_tokenizer),
            labels.shape,
            dtype=torch.long,
            device=self.devicevar.device,
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels, attention_mask

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--esm_model_version",
            type=str,
            default="facebook/esm2_t33_650M_UR50D",
            help="which version of ESM to use",
        )
        parser.add_argument(
            "--use_as_protein_encoder",
            action="store_true",
            default=False,
            help="use just to encode protein.",
        )
        parser.add_argument(
            "--use_as_mol_encoder",
            action="store_true",
            default=False,
            help="use just to encode molecule",
        )
        parser.add_argument(
            "--use_as_matching_classifier",
            action="store_true",
            default=False,
            help="use just to encode molecule-protein pair",
        )
        parser.add_argument(
            "--substrate_encoder",
            type=str,
            action=set_nox_type("model"),
            default="chemprop",
            help="Name of encoder to use for molecule",
        )
        parser.add_argument(
            "--protmol_clip_model_path",
            type=str,
            default=None,
            help="path to saved model if loading from pretrained",
        )
        parser.add_argument(
            "--mlm_probability",
            type=float,
            default=0.1,
            help="probability that a token chosen to be masked. IF chosen, 80% will be masked, 10% random, 10% original",
        )
        parser.add_argument(
            "--do_mlm_task",
            action="store_true",
            default=False,
            help="do masked language modeling",
        )
        parser.add_argument(
            "--do_matching_task",
            action="store_true",
            default=False,
            help="do molecule-protein matching",
        )
        parser.add_argument(
            "--do_reaction_node_task",
            action="store_true",
            default=False,
            help="predict if nodes in reaction graph",
        )
        parser.add_argument(
            "--gather_representations_for_matching",
            action="store_true",
            default=False,
            help="gather for creating negatives",
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
        parser.add_argument(
            "--cross_attention_frequency",
            type=int,
            default=1,
            help="interval at which to insert cross attention module",
        )
        parser.add_argument(
            "--aggregate_over_edges",
            action="store_true",
            default=False,
            help="use gat implementation.",
        )
        parser.add_argument(
            "--egnn_name",
            type=str,
            action=set_nox_type("model"),
            default="egnn_sparse_network",
            help="mlp",
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
