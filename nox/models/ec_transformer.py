import torch
import torch.nn as nn
import copy
import inspect
from typing import Union, Tuple, Any, List, Dict, Optional
from nox.utils.classes import set_nox_type
from nox.models.abstract import AbstractModel
from nox.utils.registry import register_object, get_object
from nox.utils.smiles import standardize_reaction, tokenize_smiles
from transformers import (
    EncoderDecoderModel,
    EncoderDecoderConfig,
    BertConfig,
    BertModel,
    BertLMHeadModel,
    BertTokenizer,
    AutoTokenizer,
    AutoModel,
    EsmModel,
    AutoModelForCausalLM,
    GenerationConfig,
)
from transformers.modeling_outputs import BaseModelOutput
import selfies as sf
import numpy as np


@register_object("ec_transformer", "model")
class ECTransformerDecoder(AbstractModel):
    def __init__(self, args):
        super().__init__()
        self.tokenizer = self.init_tokenizer(args)
        self.encoder = get_object(args.model_name_for_encoder, "model")(args)
        dec_config = BertConfig(
            is_decoder=True,
            add_cross_attention=True if not args.prepend_context_embedding else False,
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
        self.model = BertLMHeadModel(dec_config)
        # self.model = BertModel(dec_config)
        self.config = self.model.config
        self.args = args
        self.register_buffer("devicevar", torch.zeros(1, dtype=torch.int8))

        if args.prepend_context_embedding:
            self.position_embeddings = nn.Embedding(args.max_seq_len, args.hidden_size)
            self.register_buffer(
                "position_ids",
                torch.arange(args.max_seq_len).expand((1, -1)),
                persistent=False,
            )

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
            eos_token="[EOS]",
        )

    def forward(self, batch) -> Dict:
        ecs = [
            e.split(".") for e in batch["ec"][0]
        ]  # TODO: assumes only one ec per sample

        # Add EOS token to the end of each EC
        for i, ec in enumerate(ecs):
            ecs[i].append("[EOS]")

        decoder_input_ids = self.tokenizer(
            ecs,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
            is_split_into_words=True,
        )

        # move to device
        for k, v in decoder_input_ids.items():
            decoder_input_ids[k] = v.to(self.devicevar.device)

        return_dict = self.config.use_return_dict

        # Encode
        encoder_hidden_states = self.encoder(batch)

        if self.args.prepend_context_embedding:
            # Concatenate encoder hidden states with decoder input embeddings
            # input_embeddings = self.model.bert.embeddings(
            #     decoder_input_ids["input_ids"]
            # )  # B x L x H
            # Extract word and token type embeddings separately
            word_embeddings = self.model.bert.embeddings.word_embeddings(
                decoder_input_ids["input_ids"]
            )  # B x L x H
            token_type_ids = torch.zeros_like(
                decoder_input_ids["input_ids"]
            )  # Default token type id for decoder tokens is 0
            token_type_ids = torch.cat(
                [
                    torch.ones((token_type_ids.size(0), 1)).to(token_type_ids.device),
                    token_type_ids,
                ],
                dim=1,
            ).int()  # Adjust for prepended token
            token_type_embeddings = self.model.bert.embeddings.token_type_embeddings(
                token_type_ids
            )

            # Adjust position IDs considering the extra token
            seq_length = word_embeddings.size(1)
            position_ids = self.position_ids[
                :, : seq_length + 1
            ]  # Adjust for sequence length
            position_ids = position_ids + 1  # Shift for decoder tokens
            position_ids[:, 0] = 0  # Position ID for the prepended token

            # Compute positional embeddings
            positional_embeddings = self.position_embeddings(position_ids)

            # Combine embeddings
            input_embeddings = word_embeddings

            # TODO: add positional embeddings
            pooled_representation = encoder_hidden_states[
                "hidden"
            ]  # Assume this is ESM2 650M, B x H

            # TODO: position embeddings
            concatenated_embeddings = torch.cat(
                [pooled_representation.unsqueeze(1), input_embeddings], dim=1
            )  # B x (L+1) x H
            concatenated_embeddings += (
                positional_embeddings + token_type_embeddings
            )  # Add positional embeddings

            # # Adjust the attention mask
            extended_attention_mask = torch.cat(
                [
                    torch.ones(decoder_input_ids["input_ids"].size(0), 1).to(
                        decoder_input_ids["attention_mask"].device
                    ),
                    decoder_input_ids["attention_mask"],
                ],
                dim=1,
            )

            decoder_outputs = self.model(
                inputs_embeds=concatenated_embeddings,
                attention_mask=extended_attention_mask,
                return_dict=return_dict,
            )
            # logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            # shifted_logits = logits[:, :-2, :].contiguous()

        else:
            # # Decode
            decoder_outputs = self.model(
                input_ids=decoder_input_ids["input_ids"],
                attention_mask=decoder_input_ids["attention_mask"],
                encoder_hidden_states=encoder_hidden_states["token_hiddens"],
                encoder_attention_mask=encoder_hidden_states["mask_hiddens"].squeeze(
                    -1
                ),
                return_dict=return_dict,
            )

        logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
        # transformers/models/bert/modeling_bert.py:1233
        # we are doing next-token prediction; shift prediction scores and input ids by one
        shifted_logits = logits[:, :-1, :].contiguous()

        labels = decoder_input_ids["input_ids"].clone()
        labels[
            ~decoder_input_ids["attention_mask"].bool()
        ] = -100  # remove pad tokens from loss
        # labels = labels[:, 1:].contiguous() # TODO why is this necessary?
        labels = labels.contiguous()
        loss = torch.nn.functional.cross_entropy(
            shifted_logits.reshape(-1, self.model.config.vocab_size),
            labels.view(-1),
        )

        metric_labels = labels
        metric_logits = shifted_logits
        metric_labels = metric_labels
        output = {
            "loss": loss,
            "logit": metric_logits,
            "y": metric_labels,
        }

        if self.args.include_ec_metrics:
            predicted_token_ids = torch.argmax(metric_logits, dim=-1)
            decoded_ecs = self.tokenizer.batch_decode(predicted_token_ids)
            decoded_ecs = [ec.split(" ") for ec in decoded_ecs]

            for idx, (level, ec2index) in enumerate(self.args.ec_levels.items()):
                one_hot_vectors = []
                for ec_part in decoded_ecs:
                    num_options = len(ec2index)
                    one_hot_vector = torch.zeros(num_options)
                    truncated_ec = ".".join(ec_part[: int(level)])
                    index = ec2index.get(truncated_ec, -1)
                    if index != -1:
                        one_hot_vector[index] = 1
                    one_hot_vectors.append(one_hot_vector)
                output[f"golds_ec{level}"] = batch[f"ec{level}"].int()
                output[f"probs_ec{level}"] = (
                    torch.stack(one_hot_vectors).int().to(logits.device)
                )  # these are actually preds, just keeping name for convenience

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
        parser.add_argument(
            "--use_selfies",
            action="store_true",
            default=False,
            help="use selfies instead of SMILES for reaction",
        )
        parser.add_argument(
            "--model_name_for_encoder",
            type=str,
            action=set_nox_type("model"),
            default="fair_esm2",
            help="Name of encoder to use",
        )
        parser.add_argument(
            "--include_ec_metrics",
            action="store_true",
            default=False,
            help="whether to include EC metrics",
        )
        parser.add_argument(
            "--prepend_context_embedding",
            action="store_true",
            default=False,
            help="whether to prepend the context embedding to the input embeddings",
        )
