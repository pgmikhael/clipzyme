import torch
import torch.nn as nn
import copy
from typing import Union, Tuple, Any, List, Dict, Optional
from nox.models.abstract import AbstractModel
from nox.utils.registry import register_object
from nox.utils.smiles import standardize_reaction, tokenize_smiles

from transformers import (
    BertTokenizer,
    LongformerTokenizer,
    LongformerConfig,
    LongformerForMaskedLM,
)


@register_object("longformer", "model")
class LFormerModel(AbstractModel):
    """Longformer masked language model"""

    def __init__(self, args):
        super(LFormerModel, self).__init__()
        self.args = args

        self.hidden_aggregate_func = args.hidden_aggregate_func
        if self.hidden_aggregate_func == "cls":
            assert args.use_cls_token

        self.mlm = args.do_masked_language_model
        self.mlm_probability = args.mlm_probability

        self.tokenizer = self.init_tokenizer(args)

        config = LongformerConfig(
            max_position_embeddings=args.max_seq_len,
            vocab_size=self.tokenizer.vocab_size,
            output_hidden_states=True,
            output_attentions=True,
            num_hidden_layers = args.num_hidden_layers
        )
        self.model = LongformerForMaskedLM(config)

        # variable to easily access device
        self.register_buffer("devicevar", torch.zeros(1, dtype=torch.int8))

        # change to eval if freezing weights
        self.freeze_encoder = args.freeze_encoder
        if self.freeze_encoder:
            self.model.eval()

    def init_tokenizer(self, args):
        return LongformerTokenizer(
            vocab_file=args.vocab_path,
            merges_file=args.merges_file_path,
            do_lower_case=False,
            do_basic_tokenize=False,
            padding=True,
            truncation=True,
            model_max_length=args.max_seq_len,
        )

    def tokenize(self, list_of_text):
        raise NotImplementedError

    def forward(self, batch):
        """
        Forward pass through Longformer

        Args:
            batch (dict): dictionary of inputs. Should include "x" as key

        Returns:
            _type_: _description_
        """
        output = {}

        tokenized_inputs = self.tokenize(batch["x"])

        # from https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L607
        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = tokenized_inputs.pop("special_tokens_mask", None)

        if self.mlm:
            # mask tokens according to probability
            (
                tokenized_inputs["input_ids"],
                tokenized_inputs["labels"],
            ) = self.torch_mask_tokens(
                tokenized_inputs["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            # simple autoencoding
            labels = tokenized_inputs["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            tokenized_inputs["labels"] = labels

        # run through model
        if self.args.freeze_encoder:
            with torch.no_grad():
                result = self.model(**tokenized_inputs)
        else:
            result = self.model(**tokenized_inputs)

        # result["logit"] if label defined, (batch_size, sequence_length, config.vocab_size)
        # result["hidden_states"] tuple of (batch_size, sequence_length, hidden_size)
        # result["attentions"] tuple of (batch_size, num_heads, sequence_length, sequence_length)

        # log outputs of interest
        hidden = result["hidden_states"]
        if self.hidden_aggregate_func == "mean":
            hidden = result["hidden_states"][-1].mean(1)
        elif self.hidden_aggregate_func == "sum":
            hidden = result["hidden_states"][-1].sum(1)
        elif self.hidden_aggregate_func == "cls":
            hidden = result["hidden_states"][-1, 0]

        if self.mlm:
            masked_indices = (tokenized_inputs['input_ids'] == self.tokenizer.mask_token_id ).bool()
            output = {
                "loss": result.get("loss", None),
                "logit": result["logits"][masked_indices],
                "hidden": hidden,
                "y": tokenized_inputs["labels"][masked_indices],
            }

        else:
            output = {
                "loss": result.get("loss", None),
                "logit": result["logits"].view(-1, self.tokenizer.vocab_size),
                "hidden": hidden,
                "y": tokenized_inputs["labels"].view(-1),
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
        probability_matrix = torch.full(labels.shape, self.mlm_probability, device = self.devicevar.device)
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
            torch.bernoulli(torch.full(labels.shape, 0.8, device = self.devicevar.device)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5,device = self.devicevar.device)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long, device = self.devicevar.device
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    @staticmethod
    def add_args(parser) -> None:
        parser.add_argument(
            "--vocab_path",
            type=str,
            default=None,
            required=True,
            help="path to vocab text file required for tokenizer",
        )
        parser.add_argument(
            "--merges_file_path",
            type=str,
            default=None,
            help="path to merges file required for RoBerta tokenizer",
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
            "--use_cls_token",
            action="store_true",
            default=False,
            help="use cls token as hidden representation of sequence",
        )
        parser.add_argument(
            "--hidden_aggregate_func",
            type=str,
            default="mean",
            choices=["mean", "sum", "cls"],
            help="use cls token as hidden representation of sequence",
        )


@register_object("reaction_mlm", "model")
class ReactionMLM(LFormerModel):
    def init_tokenizer(self, args):
        return BertTokenizer(
            vocab_file=args.vocab_path,
            do_lower_case=False,
            do_basic_tokenize=False,
            sep_token=".",
            padding=True,
            truncation=True,
            model_max_length=args.max_seq_len,
            additional_special_tokens=[">>"],
            eos_token="[EOS]",
        )

    def tokenize(self, list_of_text: List[str]):
        # standardize reaction
        x = [standardize_reaction(r) for r in list_of_text]
        x = [tokenize_smiles(r, return_as_str=True) for r in x]
        if self.args.use_cls_token:
            # add [CLS] and [EOS] tokens
            x = [
                f"{self.tokenizer.cls_token} {r} {self.tokenizer.eos_token}" for r in x
            ]

        # tokenize str characters into tensor of indices with corresponding masks
        tokenized_inputs = self.tokenizer(
            x,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        # get mask for special tokens that are not masked in MLM (return_special_tokens_mask=True doesn't work for additional special tokens)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(
                toks, None, already_has_special_tokens=True
            )
            for toks in tokenized_inputs["input_ids"]
        ]

        tokenized_inputs["special_tokens_mask"] = torch.tensor(
            special_tokens_mask, dtype=torch.int64
        )

        # move to device
        for k, v in tokenized_inputs.items():
            tokenized_inputs[k] = v.to(self.devicevar.device)

        return tokenized_inputs
