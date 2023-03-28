import torch
import torch.nn as nn
import copy
import warnings
from typing import List
from nox.utils.registry import register_object, get_object
from nox.utils.classes import set_nox_type
from nox.models.abstract import AbstractModel
from nox.utils.amino_acids import AA_TO_SMILES
from torch_geometric.data import Data, HeteroData, Batch
from nox.utils.pyg import from_smiles
from nox.utils.smiles import get_rdkit_feature
from nox.models.fair_esm import FairEsm


@register_object("non_canon_net", "model")
class NonCanonicalAANet(FairEsm):
    def __init__(self, args):
        super(NonCanonicalAANet, self).__init__(args)
        self.aa_mol_encoder = get_object(args.aa_mol_encoder_name, "model")(args)

    def get_prot_smiles(self, sequences: List[str]) -> torch.tensor:
        unbatched_prot_smiles = []
        for prot in sequences:
            prots = []
            for letter in prot:
                aa = AA_TO_SMILES.get(letter, None)
                if aa is None:
                    raise Exception("AA has no smiles associated: {aa}")
                mol_datapoint = from_smiles(aa)

                try:
                    mol_datapoint.rdkit_features = torch.tensor(
                        get_rdkit_feature(aa, method=self.args.rdkit_features_name)
                    )
                except:
                    mol_datapoint.rdkit_features = torch.zeros(2048)
                    warnings.warn(
                        f"Could not get rdkit features for {aa} with method {self.args.rdkit_features_name}"
                    )

                prots.append(mol_datapoint)
            unbatched_prot_smiles.append(Batch.from_data_list(prots))
        return unbatched_prot_smiles

    def forward(self, x):
        output = {}
        if isinstance(x, list):
            pass
        elif isinstance(x, dict):
            try:
                x = x["sequence"]
            except:
                raise ValueError(
                    "FairEsm forward received dict without 'sequence' key "
                )
        elif isinstance(x, (Data, HeteroData, Batch)):
            try:
                x = x.sequence
            except:
                raise ValueError(
                    "FairEsm forward received PyG Obj without 'sequence' attr "
                )

        fair_x = self.truncate_protein(x)
        batch_labels, batch_strs, batch_tokens = self.batch_converter(fair_x)
        batch_tokens = batch_tokens.to(self.devicevar.device)

        # Each AA is encoded as a molecule
        # TODO: maybe use batch_strs?
        prot_smiles = self.get_prot_smiles(batch_strs)
        # encode prot
        prot_outs = []
        for prot in prot_smiles:
            sequence_dict = self.aa_mol_encoder(prot.to(self.devicevar.device))
            prot_outs.append(sequence_dict)

        prot_outs_hiddens = self.pad_tensor_list([s["hidden"] for s in prot_outs])

        # add CLS + EOS tokens
        prot_outs = torch.cat(
            [
                torch.zeros(
                    prot_outs_hiddens.shape[0], 1, prot_outs_hiddens.shape[-1]
                ).to(self.devicevar.device),
                prot_outs_hiddens,
                torch.zeros(
                    prot_outs_hiddens.shape[0], 1, prot_outs_hiddens.shape[-1]
                ).to(self.devicevar.device),
            ],
            dim=1,
        )

        # ESM transformer is run on the encoded molecules
        if self.args.freeze_esm:
            with torch.no_grad():
                result = self.esm_forward(
                    prot_outs,
                    batch_tokens,
                    repr_layers=[self.repr_layer],
                    return_contacts=False,
                )
        else:
            result = self.esm_forward(
                prot_outs,
                batch_tokens,
                repr_layers=[self.repr_layer],
                return_contacts=False,
            )

        # Generate per-sequence representations via averaging
        if self.use_cls_token:
            output["hidden"] = result["representations"][self.repr_layer][0]
        else:
            # remove cls, eos, and padding embeddings
            sequence_mask = torch.ne(batch_tokens, self.alphabet.cls_idx).long()
            sequence_mask *= torch.ne(batch_tokens, self.alphabet.eos_idx).long()
            sequence_mask *= torch.ne(batch_tokens, self.alphabet.padding_idx).long()
            sequence_mask = sequence_mask.unsqueeze(-1)
            # remove cls and eos tokens
            output["hidden"] = (
                result["representations"][self.repr_layer] * sequence_mask
            ).sum(1) / sequence_mask.sum(1)

        output["tokens"] = batch_tokens
        output["token_hiddens"] = result["representations"][self.repr_layer]
        output["mask_hiddens"] = sequence_mask
        return output

    def pad_tensor_list(self, tensor_list):
        """Pad a list of tensors to the same length"""
        max_len = max([t.shape[0] for t in tensor_list])
        padded_tensor_list = []
        for t in tensor_list:
            padded_tensor_list.append(
                torch.cat(
                    [
                        t,
                        torch.zeros(max_len - t.shape[0], t.shape[1]).to(
                            self.devicevar.device
                        ),
                    ]
                )
            )
        return torch.stack(padded_tensor_list)

    def esm_forward(
        self, x, tokens, repr_layers=[], need_head_weights=False, return_contacts=False
    ):
        """
        Identical to ESM forward method, but with alternative x input
        """
        if return_contacts:
            need_head_weights = True

        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.model.padding_idx)  # B, T

        # removed from original esm forward method
        # x = self.model.embed_scale * self.model.embed_tokens(tokens)

        if self.args.add_to_embeddings:
            x += self.model.embed_scale * self.model.embed_tokens(tokens)

        if self.model.token_dropout:
            x.masked_fill_((tokens == self.model.mask_idx).unsqueeze(-1), 0.0)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (tokens == self.model.mask_idx).sum(-1).to(
                x.dtype
            ) / src_lengths
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            attn_weights = []

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None

        for layer_idx, layer in enumerate(self.model.layers):
            x, attn = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
            )
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            if need_head_weights:
                # (H, B, T, T) => (B, H, T, T)
                attn_weights.append(attn.transpose(1, 0))

        x = self.model.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        # last hidden representation should have layer norm applied
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x
        x = self.model.lm_head(x)

        result = {"logits": x, "representations": hidden_representations}
        if need_head_weights:
            # attentions: B x L x H x T x T
            attentions = torch.stack(attn_weights, 1)
            if padding_mask is not None:
                attention_mask = 1 - padding_mask.type_as(attentions)
                attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(
                    2
                )
                attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions
            if return_contacts:
                contacts = self.model.contact_head(tokens, attentions)
                result["contacts"] = contacts

        return result

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        super(NonCanonicalAANet, NonCanonicalAANet).add_args(parser)
        parser.add_argument(
            "--aa_mol_encoder_name",
            type=str,
            action=set_nox_type("model"),
            default="chemprop",
            help="Name of encoder to use",
        )
        parser.add_argument(
            "--add_to_embeddings",
            action="store_true",
            default=False,
            help="Add the molecule embeddings to the AA embeddings",
        )


# this is kind of rubbish for debugging - can ignore
@register_object("non_canon_net_simple", "model")
class NonCanonicalAANetSimple(AbstractModel):
    def __init__(self, args):
        super(NonCanonicalAANetSimple, self).__init__()
        self.args = args
        self.mol_encoder = get_object(args.mol_encoder_name, "model")(args)

    def forward(self, batch=None):
        output = {}
        if isinstance(batch, list):
            pass
        elif isinstance(batch, dict):
            try:
                x = batch["sequence"]
            except:
                raise ValueError(
                    "SimpleNonCanon forward received dict without 'sequence' key "
                )
        elif isinstance(batch, (Data, HeteroData, Batch)):
            try:
                x = batch.sequence
            except:
                raise ValueError(
                    "SimpleNonCanon forward received PyG Obj without 'sequence' attr "
                )

        # turn AA seq to molecules
        prot_smiles = self.get_prot_smiles(batch_strs)
        # encode prot
        prot_outs = []
        for prot in prot_smiles:
            sequence_dict = self.aa_mol_encoder(prot.to(self.devicevar.device))
            prot_outs.append(sequence_dict)

        prot_outs_hiddens = self.pad_tensor_list([s["hidden"] for s in prot_outs])
        prot_outs = torch.mean(prot_outs_hiddens, dim=0)

        output = {
            "sequence_hidden": prot_outs,
        }
        return output

    def pad_tensor_list(self, tensor_list):
        """Pad a list of tensors to the same length"""
        max_len = max([t.shape[0] for t in tensor_list])
        padded_tensor_list = []
        for t in tensor_list:
            padded_tensor_list.append(
                torch.cat(
                    [
                        t,
                        torch.zeros(max_len - t.shape[0], t.shape[1]).to(
                            self.devicevar.device
                        ),
                    ]
                )
            )
        return torch.stack(padded_tensor_list)

    def get_prot_smiles(self, sequences: List[str]) -> torch.tensor:
        unbatched_prot_smiles = []
        for prot in sequences:
            prots = []
            for letter in prot:
                aa = AA_TO_SMILES.get(letter, None)
                if aa is None:
                    raise Exception("AA has no smiles associated: {aa}")
                mol_datapoint = from_smiles(aa)
                mol_datapoint.rdkit_features = torch.tensor(
                    get_rdkit_feature(aa, method=self.args.rdkit_features_name)
                )
                prots.append(mol_datapoint)
            unbatched_prot_smiles.append(Batch.from_data_list(prots))
        return unbatched_prot_smiles

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--mol_encoder_name",
            type=str,
            action=set_nox_type("model"),
            default="chemprop",
            help="Name of molecular encoder to use",
        )
        parser.add_argument(
            "--prot_encoder_name",
            type=str,
            action=set_nox_type("model"),
            default="fair_esm2",
            help="Name of protein encoder to use",
        )
