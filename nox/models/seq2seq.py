import torch
import torch.nn as nn
import copy
from typing import Union, Tuple, Any, List, Dict, Optional
from nox.utils.classes import set_nox_type
from nox.models.abstract import AbstractModel
from nox.utils.registry import register_object, get_object
from nox.utils.smiles import standardize_reaction, tokenize_smiles
from transformers import EncoderDecoderModel, LongformerConfig, BertTokenizer
from transformers.modeling_outputs import BaseModelOutput


@register_object("reaction_encoder", "model")
class ReactionEncoder(AbstractModel):
    def __init__(self, args):
        super().__init__()

        architecture_config = self.get_transformer_model(args)

        self.tokenizer = self.init_tokenizer(args)

        self.model = EncoderDecoderModel.from_encoder_decoder_configs(
            architecture_config, architecture_config
        )
        self.config = self.model.config

    def get_transformer_model(self, args):
        if args.transformer_model == "longformer":
            config = LongformerConfig(
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
        else:
            raise NotImplementedError
        return config

    @staticmethod
    def init_tokenizer(args):
        return BertTokenizer(
            vocab_file=args.vocab_path,
            do_lower_case=False,
            do_basic_tokenize=False,
            sep_token=".",
            padding=True,
            truncation=True,
            model_max_length=args.max_seq_len,
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

        encoder_input_ids = self.tokenize(batch["reactants"], self.tokenizer, self.args)
        decoder_input_ids = self.tokenize(batch["products"], self.tokenizer, self.args)

        # move to device
        for k, v in encoder_input_ids.items():
            encoder_input_ids[k] = v.to(self.devicevar.device)

        for k, v in decoder_input_ids.items():
            decoder_input_ids[k] = v.to(self.devicevar.device)

        # from https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L607
        # If special token mask has been preprocessed, pop it from the dict.
        # special_tokens_mask = tokenized_inputs.pop("special_tokens_mask", None)

        # output = self.model(
        #     input_ids=encoder_input_ids,
        #     decoder_input_ids=decoder_input_ids,
        #     output_hidden_states=True,
        #     return_dict=True,
        # )

        # return output

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if encoder_outputs is None:
            encoder_outputs = self.model.encoder(input_ids=encoder_input_ids)

        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)

        encoder_hidden_states = encoder_outputs[0]

        # optionally project encoder_hidden_states
        if (
            self.model.encoder.config.hidden_size
            != self.model.decoder.config.hidden_size
            and self.model.decoder.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_states = self.model.enc_to_dec_proj(encoder_hidden_states)

        # Decode
        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=return_dict,
        )

        logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
        loss = torch.nn.functinal.cross_entropy_loss(
            logits.reshape(-1, self.decoder.config.vocab_size),
            decoder_input_ids.view(-1),
        )

        output = {
            "logits": decoder_outputs.logits,
            "decoder_hidden_states": decoder_outputs.hidden_states,
            "past_key_values": decoder_outputs.past_key_values,
            "decoder_hidden_states": decoder_outputs.hidden_states,
            "decoder_attentions": decoder_outputs.attentions,
            "cross_attentions": decoder_outputs.cross_attentions,
            "encoder_last_hidden_state": encoder_outputs.last_hidden_state,
            "encoder_hidden_states": encoder_outputs.hidden_states,
            "encoder_attentions": encoder_outputs.attentions,
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
        parser.add_argument(
            "--longformer_model_path",
            type=str,
            default=None,
            help="path to saved model if loading from pretrained",
        )


@register_object("reaction_encoder", "model")
class EnzymaticReactionEncoder(ReactionEncoder):
    def __init__(self, args):
        super().__init__(args)
        self.protein_model = get_object(args, "model")(args)
        self.append_enzyme_to_hiddens = args.append_enzyme_to_hiddens

    def forward(self, batch) -> Dict:

        protein_output = self.protein_model(batch)

        encoder_input_ids = self.tokenize(batch["reactants"], self.tokenizer, self.args)
        decoder_input_ids = self.tokenize(batch["products"], self.tokenizer, self.args)

        # move to device
        for k, v in encoder_input_ids.items():
            encoder_input_ids[k] = v.to(self.devicevar.device)

        for k, v in decoder_input_ids.items():
            decoder_input_ids[k] = v.to(self.devicevar.device)

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if self.append_enzyme_to_hiddens:
            # append to embeddings of reactants
            embeddings = self.model.encoder.embeddings(input_ids=encoder_input_ids)
            embeddings_w_protein = torch.cat(
                [protein_output["hidden"], embeddings], dim=1
            )
            encoder_outputs = self.model.encoder.model(embeddings_w_protein)
        else:
            # get reactant embeddings
            encoder_outputs = self.model.encoder(input_ids=encoder_input_ids)

        encoder_hidden_states = encoder_outputs[0]

        if self.append_enzyme_to_hiddens:
            encoder_hidden_states = torch.cat(
                [protein_output["hidden"], encoder_hidden_states], dim=1
            )

        # optionally project encoder_hidden_states
        if (
            self.model.encoder.config.hidden_size
            != self.model.decoder.config.hidden_size
            and self.model.decoder.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_states = self.model.enc_to_dec_proj(encoder_hidden_states)

        # Decode
        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=return_dict,
        )

        logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
        loss = torch.nn.functinal.cross_entropy_loss(
            logits.reshape(-1, self.decoder.config.vocab_size),
            decoder_input_ids.view(-1),
        )

        output = {
            "logits": decoder_outputs.logits,
            "decoder_hidden_states": decoder_outputs.hidden_states,
            "past_key_values": decoder_outputs.past_key_values,
            "decoder_hidden_states": decoder_outputs.hidden_states,
            "decoder_attentions": decoder_outputs.attentions,
            "cross_attentions": decoder_outputs.cross_attentions,
            "encoder_last_hidden_state": encoder_outputs.last_hidden_state,
            "encoder_hidden_states": encoder_outputs.hidden_states,
            "encoder_attentions": encoder_outputs.attentions,
        }
        return output

    @staticmethod
    def add_args(parser) -> None:
        super(EnzymaticReactionEncoder, EnzymaticReactionEncoder).add_args(parser)
        parser.add_argument(
            "--enzyme_model",
            action=set_nox_type("model"),
            default="fair_esm",
            help="protein_model",
        )
        parser.add_argument(
            "--append_enzyme_to_hiddens",
            action="store_true",
            default=False,
            help="whether to append enzyme to hidden states. otherwise appended to inputs",
        )
