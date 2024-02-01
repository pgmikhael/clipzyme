import torch
from typing import List, Dict
from clipzyme.models.abstract import AbstractModel
from clipzyme.utils.registry import register_object
from clipzyme.utils.smiles import standardize_reaction, tokenize_smiles
from transformers import (
    BertConfig,
    BertModel,
    BertTokenizer,
)


@register_object("full_reaction_encoder", "model")
class FullReactionEncoder(AbstractModel):
    def __init__(self, args):
        super().__init__()

        self.tokenizer = self.init_tokenizer(args)

        enc_config = BertConfig(
            max_position_embeddings=args.max_seq_len,
            vocab_size=self.tokenizer.vocab_size,
            output_hidden_states=True,
            output_attentions=True,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            embedding_size=args.embedding_size,
            num_attention_heads=args.num_heads,
            num_hidden_layers=args.num_hidden_layers,
        )

        self.model = BertModel(enc_config)
        self.config = self.model.config
        self.args = args
        self.register_buffer("devicevar", torch.zeros(1, dtype=torch.int8))

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
            cls_token="[CLS]",
            eos_token="[EOS]",
        )

    @staticmethod
    def tokenize(list_of_text: List[str], tokenizer, args):
        # standardize reaction
        x = [standardize_reaction(r) for r in list_of_text]
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

    def forward(self, batch) -> Dict:
        # tokenize full reaction
        encoder_input_ids = self.tokenize(batch["reaction"], self.tokenizer, self.args)

        # move to device
        for k, v in encoder_input_ids.items():
            encoder_input_ids[k] = v.to(self.devicevar.device)

        return_dict = self.config.use_return_dict

        encoder_outputs = self.model(
            input_ids=encoder_input_ids["input_ids"],
            attention_mask=encoder_input_ids["attention_mask"],
        )

        encoder_hidden_states = encoder_outputs[0]

        output = {
            "encoder_output": encoder_outputs["pooler_output"]
            # "encoder_hidden_states": encoder_outputs.hidden_states,
            # "encoder_attentions": encoder_outputs.attentions,
        }
        return output

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
            "--embedding_size",
            type=int,
            default=128,
            help="maximum length allowed for the input sequence",
        )
        parser.add_argument(
            "--num_heads",
            type=int,
            default=8,
            help="maximum length allowed for the input sequence",
        )
        parser.add_argument(
            "--use_cls_token",
            action="store_true",
            default=False,
            help="use cls token as hidden representation of sequence",
        )
