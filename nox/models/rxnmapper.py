"""
Core RXN Attention Mapper module.
Adapted from: https://github.com/rxn4chemistry/rxnmapper/blob/main/rxnmapper/core.py
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pkg_resources
import torch
import torch.nn as nn
from rxn.chemutils.reaction_equation import ReactionEquation
from rxn.chemutils.reaction_smiles import (
    ReactionFormat,
    determine_format,
    parse_any_reaction_smiles,
    to_reaction_smiles,
)
from transformers import AlbertModel, BertModel, RobertaModel, AlbertForMaskedLM

from rxnmapper.attention import AttentionScorer
from rxnmapper.smiles_utils import generate_atom_mapped_reaction_atoms, process_reaction
from rxnmapper.tokenization_smiles import SmilesTokenizer

MODEL_TYPE_DICT = {"bert": BertModel, "albert": AlbertModel, "roberta": RobertaModel, "albertlm": AlbertForMaskedLM}

from nox.utils.registry import register_object
from nox.models.abstract import AbstractModel

_logger = logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())


@register_object("rxn_mapper", "model")
class RXNMapper(AbstractModel):
    """Wrap the RXNMapper model, corresponding tokenizer, and attention scoring algorithms.
    Maps product atoms to reactant atoms using the attention weights
    """

    def __init__(self, args):
        super(RXNMapper, self).__init__()
        config = {}

        # Config takes "model_path", "model_type", "attention_multiplier", "head", "layers"
        self.model_path = config.get(
            "model_path",
            pkg_resources.resource_filename(
                "rxnmapper", "models/transformers/albert_heads_8_uspto_all_1310k"
            ),
        )

        self.model_type = config.get("model_type", "albertlm")
        self.attention_multiplier = config.get("attention_multiplier", 90.0)
        self.head = config.get("head", 5)
        self.layers = config.get("layers", [10])

        self.logger = _logger
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        self.model.albert.encoder.albert_layer_groups[0].albert_layers[
            0
        ].activation == nn.Identity()  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model.to(self.device)

        # M-LM
        self.mlm = args.do_masked_language_model
        self.mlm_probability = args.mlm_probability

    def _load_model_and_tokenizer(self) -> Tuple:
        """
        Load transformer and tokenizer model.
        Returns:
            Tuple: containing model and tokenizer
        """
        model_class = MODEL_TYPE_DICT[self.model_type]
        model = model_class.from_pretrained(
            self.model_path,
            output_attentions=True,
            output_past=False,
            output_hidden_states=False,
        )

        vocab_path = None

        if os.path.exists(os.path.join(self.model_path, "vocab.txt")):
            vocab_path = os.path.join(self.model_path, "vocab.txt")

        tokenizer = SmilesTokenizer(
            vocab_path, max_len=model.config.max_position_embeddings
        )
        return (model, tokenizer)

    def forward(self,batch):
        """Extract desired attentions from a given batch of reactions.
        Args:
            rxn_smiles_list: List of reactions to mape
            force_layer: If given, override the default layer used for RXNMapper
            force_head: If given, override the default head used for RXNMapper
        """

        rxn_smiles_list = batch["x"]

        reactions = [parse_any_reaction_smiles(rxn) for rxn in rxn_smiles_list]
        reactions = [process_reaction(reaction) for reaction in reactions]

        # The transformer has been trained on the format containing tildes.
        # This means that we must convert to that format for use with the model.
        reactions = [
            to_reaction_smiles(
                reaction, reaction_format=ReactionFormat.STANDARD_WITH_TILDE
            )
            for reaction in reactions
        ]

        encoded_ids = self.tokenizer.batch_encode_plus(
            reactions,
            padding=True,
            return_tensors="pt",
        )


        # get mask for special tokens that are not masked in MLM (return_special_tokens_mask=True doesn't work for additional special tokens)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(
                toks, None, already_has_special_tokens=True
            )
            for toks in encoded_ids["input_ids"]
        ]

        encoded_ids["special_tokens_mask"] = torch.tensor(
            special_tokens_mask, dtype=torch.int64
        )

        parsed_input = {k: v.to(self.device) for k, v in encoded_ids.items()}

        # from https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L607
        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = parsed_input.pop("special_tokens_mask", None)

        if self.mlm:
            # mask tokens according to probability
            (
                parsed_input["input_ids"],
                parsed_input["labels"],
            ) = self.torch_mask_tokens(
                parsed_input["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            # simple autoencoding
            labels = parsed_input["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            parsed_input["labels"] = labels

        with torch.no_grad():
            result = self.model(**parsed_input)


        if self.mlm:
            masked_indices = (parsed_input['input_ids'] == self.tokenizer.mask_token_id ).bool()
            output = {
                "loss": result.get("loss", None),
                "logit": result["logits"][masked_indices],
                # "hidden": hidden,
                "y": parsed_input["labels"][masked_indices],
            }

        else:
            output = {
                "loss": result.get("loss", None),
                "logit": result["logits"].view(-1, self.tokenizer.vocab_size),
                # "hidden": hidden,
                "y": parsed_input["labels"].view(-1),
            }

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
        probability_matrix = torch.full(labels.shape, self.mlm_probability, device = self.device)
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
            torch.bernoulli(torch.full(labels.shape, 0.8, device = self.device)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5,device = self.device)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long, device = self.device
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


    @staticmethod
    def add_args(parser) -> None:
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