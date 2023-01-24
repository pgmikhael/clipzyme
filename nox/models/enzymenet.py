import torch
import torch.nn as nn
import copy
from nox.utils.registry import register_object, get_object
from nox.utils.classes import set_nox_type
from nox.models.abstract import AbstractModel


@register_object("enzyme_substrate_model", "model")
class EnzymeSubstrateScore(AbstractModel):
    def __init__(self, args):
        super(EnzymeSubstrateScore, self).__init__()

        self.args = args
        self.protein_encoder = get_object(args.protein_encoder_name, "model")(args)
        self.substrate_encoder = get_object(args.substrate_encoder_name, "model")(args)
        self.mlp = get_object(args.protein_substrate_aggregator, "model")(args)
        if args.activation_name is not None:
            self.activation = getattr(torch.nn.functional, args.activation_name)

    def forward(self, batch=None):
        output = {}
        sequence_dict = self.protein_encoder(batch["sequence"])
        substrate_dict = self.substrate_encoder(batch["mol"])
        hidden = torch.cat((sequence_dict["hidden"], substrate_dict["hidden"]), dim=1)
        mlp_dict = self.mlp({"x": hidden})
        output = {
            "sequence_hidden": sequence_dict["hidden"],
            "substrate_hidden": substrate_dict["hidden"],
        }
        output.update(mlp_dict)
        if self.args.activation_name is not None:
            output["logit"] = self.activation(output["logit"])
        return output

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--protein_encoder_name",
            type=str,
            action=set_nox_type("model"),
            default="fair_esm",
            help="Name of encoder to use",
        )
        parser.add_argument(
            "--substrate_encoder_name",
            type=str,
            action=set_nox_type("model"),
            default="chemprop",
            help="Name of encoder to use",
        )
        parser.add_argument(
            "--protein_substrate_aggregator",
            type=str,
            action=set_nox_type("model"),
            default="chemprop",
            help="Name of encoder to use",
        )
        parser.add_argument(
            "--activation_name",
            type=str,
            default=None,
            help="type of activation to be applied on logits",
        )


@register_object("enzyme_active_site_predictor", "model")
class EnzymeActiveSiteModel(AbstractModel):
    def __init__(self, args):
        super(EnzymeActiveSiteModel, self).__init__()

        self.args = args
        classifier_args = copy.deepcopy(args)
        self.protein_encoder = get_object(args.protein_encoder_name, "model")(args)
        if args.reaction_encoder_name is not None:
            self.reaction_encoder = get_object(args.reaction_encoder_name, "model")(
                args
            )
            classifier_args.mlp_input_dim = args.mlp_input_dim + args.hidden_size
        self.mlp = get_object(args.classifier_name, "model")(classifier_args)

    def forward(self, batch=None):
        batch_size = len(batch["sequence"])
        seq_len = [len(p) for p in batch["sequence"]]
        sequence_dict = self.protein_encoder(batch["sequence"])
        hidden = sequence_dict["token_hiddens"][:, 1:-1]  # B, seq_len, hidden_dim

        if hasattr(self, "reaction_encoder"):
            rxn_dict = self.reaction_encoder({"x": batch["reaction"]})
            rxn_h = rxn_dict["hidden"].unsqueeze(1)  # B, 1, hidden_dim
            rxn_h = torch.repeat_interleave(rxn_h, hidden.shape[1], dim=1)
            hidden = torch.concat([hidden, rxn_h], dim=-1)

        output = self.mlp({"x": hidden})  # B, seq_len, num_classes

        labels = batch["residue_mask"]
        residue_mask = sequence_dict["mask_hiddens"][:, 1:-1]
        # cross entropy will ignore the padded residues (ignore_index=-100)
        labels[~residue_mask.squeeze(-1).bool()] = -100
        batch["y"] = labels

        return output

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--protein_encoder_name",
            type=str,
            action=set_nox_type("model"),
            default="fair_esm",
            help="Name of encoder to use",
        )
        parser.add_argument(
            "--reaction_encoder_name",
            type=str,
            action=set_nox_type("model"),
            default=None,
            help="Name of encoder to use",
        )
        parser.add_argument(
            "--classifier_name",
            type=str,
            action=set_nox_type("model"),
            default="mlp_classifier",
            help="Name of encoder to use",
        )


@register_object("enzyme_active_site_predictor_attention", "model")
class EnzymeActiveSiteAttentionModel(EnzymeActiveSiteModel):
    def forward(self, batch=None):
        batch_size = len(batch["sequence"])
        seq_len = [len(p) for p in batch["sequence"]]
        sequence_dict = self.protein_encoder(batch["sequence"])
        """
        originally attentions = batch_size, layers, heads, seqlen, seqlen
        if using contacts then:
        then attentions = batch_size, layers * heads, seqlen, seqlen
        then (permute) attentions = batch_size, seqlen, seqlen, layers * heads (0, 2, 3, 1)
        then self.activation(self.regression(attentions).squeeze(3)) = batch_size, seqlen, seqlen
        note sigmoid is applied in the activation function, so we have 0<=a_b,i,j<=1 for b batch, i,j tokens
        """
        attentions = sequence_dict["attentions"]  # B, layers, heads, seqlen, seqlen
        tokens = sequence_dict["tokens"]
        self.prepend_bos = self.protein_encoder.alphabet.prepend_bos
        self.append_eos = self.protein_encoder.alphabet.append_eos
        self.eos_idx = self.protein_encoder.alphabet.eos_idx
        # remove eos token attentions
        if self.append_eos:
            eos_mask = tokens.ne(self.eos_idx).to(attentions)
            eos_mask = eos_mask.unsqueeze(1) * eos_mask.unsqueeze(2)
            attentions = attentions * eos_mask[:, None, None, :, :]
            attentions = attentions[..., :-1, :-1]
        # remove cls token attentions
        if self.prepend_bos:
            attentions = attentions[..., 1:, 1:]

        batch_size, layers, heads, seqlen, _ = attentions.size()
        attentions = attentions.view(
            batch_size, layers * heads, seqlen, seqlen
        )  # features: B x C x T x T
        attentions = attentions.permute(0, 2, 3, 1)  # features: B x T x T x C
        # aggregate across each token
        attentions = attentions.sum(dim=2)  # features: B x T x C

        output = self.mlp({"x": attentions})  # B x T x num_classes

        residue_mask = sequence_dict["mask_hiddens"][:, 1:-1]
        labels = batch["residue_mask"]
        labels = labels[:, :residue_mask.shape[1]].contiguous() # if truncating protein, truncates labels
        # cross entropy will ignore the padded residues (ignore_index=-100)
        labels[~residue_mask.squeeze(-1).bool()] = -100
        batch["y"] = labels

        return output

    @staticmethod
    def set_args(args) -> None:
        super(EnzymeActiveSiteModel, EnzymeActiveSiteModel).set_args(args)
        args.mlp_input_dim = 20 * 12


@register_object("enzyme_active_site_predictor_contact_maps", "model")
class EnzymeActiveSiteAttentionModel(EnzymeActiveSiteModel):
    def forward(self, batch=None):
        batch_size = len(batch["sequence"])
        seq_len = [len(p) for p in batch["sequence"]]
        sequence_dict = self.protein_encoder(batch["sequence"])
        """
        originally attentions = batch_size, layers, heads, seqlen, seqlen
        if using contacts then:
        then attentions = batch_size, layers * heads, seqlen, seqlen
        then (permute) attentions = batch_size, seqlen, seqlen, layers * heads (0, 2, 3, 1)
        then self.activation(self.regression(attentions).squeeze(3)) = batch_size, seqlen, seqlen
        note sigmoid is applied in the activation function, so we have 0<=a_b,i,j<=1 for b batch, i,j tokens
        """
        contacts = sequence_dict["contacts"] # B, (batch_tokens - 2), (batch_tokens - 2)

        contacts = contacts.sum(dim=2)  # features: B x T

        output = self.mlp({"x": attentions})  # B x T x num_classes

        residue_mask = sequence_dict["mask_hiddens"][:, 1:-1]
        labels = batch["residue_mask"]
        labels = labels[:, :residue_mask.shape[1]].contiguous() # if truncating protein, truncates labels
        # cross entropy will ignore the padded residues (ignore_index=-100)
        labels[~residue_mask.squeeze(-1).bool()] = -100
        batch["y"] = labels

        return output

    @staticmethod
    def set_args(args) -> None:
        super(EnzymeActiveSiteModel, EnzymeActiveSiteModel).set_args(args)
        args.mlp_input_dim = 20 * 12
