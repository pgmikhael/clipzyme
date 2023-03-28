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
    BertTokenizer,
    AutoTokenizer,
    AutoModel,
    EsmModel,
    AutoModelForCausalLM,
)
from transformers.modeling_outputs import BaseModelOutput


@register_object("reaction_encoder", "model")
class ReactionEncoder(AbstractModel):
    def __init__(self, args):
        super().__init__()

        self.tokenizer = self.init_tokenizer(args)

        econfig, dconfig = self.get_transformer_model(args)
        config = EncoderDecoderConfig.from_encoder_decoder_configs(econfig, dconfig)
        self.model = EncoderDecoderModel(config=config)
        self.config = self.model.config
        self.args = args
        self.register_buffer("devicevar", torch.zeros(1, dtype=torch.int8))

        self.generation_config = self.make_generation_config(args)

    def make_generation_config(self, args):
        generate_args = inspect.signature(self.model.generate).parameters
        args_dict = vars(args)
        gen_config = {}

        for k in generate_args:
            if f"generation_{k}" in args_dict:
                gen_config[k] = args_dict[f"generation_{k}"]

        return gen_config

    def get_transformer_model(self, args):
        if args.transformer_model == "bert":
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
            dec_config = BertConfig(
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
        return enc_config, dec_config

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
        x = [standardize_reaction(r + ">>")[:-2] for r in list_of_text]
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

        if self.args.predict:
            predictions = self.generate(batch)
            return {
                "preds": predictions,
                "golds": batch["products"],
            }

        encoder_input_ids = self.tokenize(batch["reactants"], self.tokenizer, self.args)
        decoder_input_ids = self.tokenize(batch["products"], self.tokenizer, self.args)

        # move to device
        for k, v in encoder_input_ids.items():
            encoder_input_ids[k] = v.to(self.devicevar.device)

        for k, v in decoder_input_ids.items():
            decoder_input_ids[k] = v.to(self.devicevar.device)

        return_dict = self.config.use_return_dict

        encoder_outputs = self.model.encoder(
            input_ids=encoder_input_ids["input_ids"],
            attention_mask=encoder_input_ids["attention_mask"],
        )

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
            input_ids=decoder_input_ids["input_ids"],
            attention_mask=decoder_input_ids["attention_mask"],
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_input_ids["attention_mask"],
            return_dict=return_dict,
        )

        labels = decoder_input_ids["input_ids"].clone()
        labels[
            ~decoder_input_ids["attention_mask"].bool()
        ] = -100  # remove pad tokens from loss
        logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
        # transformers/models/bert/modeling_bert.py:1233
        # we are doing next-token prediction; shift prediction scores and input ids by one
        shifted_logits = logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shifted_logits.reshape(-1, self.model.decoder.config.vocab_size),
            labels.view(-1),
        )

        metric_labels = labels  # .view(-1)
        metric_logits = shifted_logits  # .reshape(-1, self.model.decoder.config.vocab_size)[metric_labels!= -100]
        metric_labels = metric_labels  # [ metric_labels!= -100]
        output = {
            "loss": loss,
            "logit": metric_logits,
            "y": metric_labels,
            # "decoder_hidden_states": decoder_outputs.hidden_states,
            # "past_key_values": decoder_outputs.past_key_values,
            # "decoder_hidden_states": decoder_outputs.hidden_states,
            # "decoder_attentions": decoder_outputs.attentions,
            # "cross_attentions": decoder_outputs.cross_attentions,
            # "encoder_last_hidden_state": encoder_outputs.last_hidden_state,
            # "encoder_hidden_states": encoder_outputs.hidden_states,
            # "encoder_attentions": encoder_outputs.attentions,
        }
        return output

    def generate(self, batch):
        """Applies auto-regressive generation for reaction prediction
        Usage: https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin

        Args:
            batch (dict): dictionary with input reactants
            decoder_start_token_id = 2,
            bos_token_id = 2,
            max_new_tokens=100
        """
        bos_id = self.tokenizer.cls_token_id
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id

        encoder_input_ids = self.tokenize(batch["reactants"], self.tokenizer, self.args)
        for k, v in encoder_input_ids.items():
            encoder_input_ids[k] = v.to(self.devicevar.device)

        generated_ids = self.model.generate(
            input_ids=encoder_input_ids["input_ids"],
            decoder_start_token_id=bos_id,
            bos_token_id=bos_id,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            attention_mask=encoder_input_ids["attention_mask"],
            output_scores=True,
            return_dict_in_generate=True,
            **self.generation_config,
        )
        generated_samples = self.tokenizer.batch_decode(
            generated_ids.sequences, skip_special_tokens=True
        )
        generated_samples = [s.replace(" ", "") for s in generated_samples]
        beam_search_n = self.generation_config["num_return_sequences"]
        batch_samples = [
            generated_samples[i : i + beam_search_n]
            for i in range(0, len(generated_samples), beam_search_n)
        ]
        return batch_samples

    @staticmethod
    def add_args(parser) -> None:
        parser.add_argument(
            "--transformer_model",
            type=str,
            default="bert",
            help="name of backbone model",
        )
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
        parser.add_argument(
            "--generation_max_new_tokens",
            type=int,
            default=200,
            help="The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.",
        )
        parser.add_argument(
            "--generation_num_beams",
            type=int,
            default=1,
            help="Number of beams for beam search. 1 means no beam search.",
        )
        parser.add_argument(
            "--generation_num_return_sequences",
            type=int,
            default=1,
            help="The number of independently computed returned sequences for each element in the batch. <= num_beams",
        )
        parser.add_argument(
            "--generation_num_beam_groups",
            type=int,
            default=1,
            help="Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams. [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details",
        )
        parser.add_argument(
            "--generation_do_sample",
            action="store_true",
            default=False,
            help="Whether or not to use sampling ; use greedy decoding otherwise.",
        )
        parser.add_argument(
            "--generation_early_stopping",
            action="store_true",
            default=False,
            help="Whether to stop the beam search when at least `num_beams` sentences are finished per batch or not",
        )
        parser.add_argument(
            "--generation_renormalize_logits",
            action="store_true",
            default=False,
            help=" Whether to renormalize the logits after applying all the logits processors or warpers (including the custom ones). It's highly recommended to set this flag to `True` as the search algorithms suppose the score logits are normalized but some logit processors or warpers break the normalization.",
        )


@register_object("enzymatic_reaction_encoder", "model")
class EnzymaticReactionEncoder(ReactionEncoder):
    def __init__(self, args):
        super().__init__(args)
        if args.enzyme_model is not None:
            self.protein_model = get_object(args.enzyme_model, "model")(args)
        else:
            assert args.precomputed_esm_features_dir is not None

        self.append_enzyme_to_hiddens = args.append_enzyme_to_hiddens
        self.protein_representation_key = args.protein_representation_key
        if args.hidden_size != args.protein_feature_dim:
            self.protein_fc = nn.Linear(args.protein_feature_dim, args.hidden_size)

    def encode_reactants_and_enzyme(self, batch, encoder_input_ids):
        # pass sequences through protein model or use precomputed hiddens
        if hasattr(self, "protein_model"):
            protein_output = self.protein_model(batch)
            protein_attention = torch.ne(
                protein_output["tokens"], self.protein_model.alphabet.padding_idx
            ).long()
        else:
            protein_output = batch
            protein_attention = torch.zeros(
                len(batch["sequence"]), max(batch["protein_len"])
            ).to(self.devicevar.device)
            for i, length in enumerate(batch["protein_len"]):
                protein_attention[i, :length] = 1

        protein_embeds = protein_output[self.protein_representation_key]
        if len(protein_embeds.shape) == 2:
            protein_embeds = protein_embeds.unsqueeze(1).contiguous()

        # project to hidden_dim if necessary
        if hasattr(self, "protein_fc"):
            protein_embeds = self.protein_fc(protein_embeds)

        # combine protein - reactants attention mask
        if self.protein_representation_key == "token_hiddens":
            protein_reactants_attention_mask = torch.cat(
                [protein_attention, encoder_input_ids["attention_mask"]], dim=-1
            )
            # token id types (protein vs reactants)
            token_type_ids = torch.cat(
                [protein_attention * 0, encoder_input_ids["attention_mask"]], dim=-1
            ).long()
        elif self.protein_representation_key == "hidden":
            protein_reactants_attention_mask = torch.cat(
                [protein_attention[:, :1], encoder_input_ids["attention_mask"]], dim=-1
            )
            # token id types (protein vs reactants)
            token_type_ids = torch.cat(
                [protein_attention[:, :1] * 0, encoder_input_ids["attention_mask"]],
                dim=-1,
            ).long()

        if not self.append_enzyme_to_hiddens:
            # append to embeddings of reactants
            embeddings = self.model.encoder.embeddings(
                input_ids=encoder_input_ids["input_ids"],
                token_type_ids=None,
                past_key_values_length=0,
            )
            embeddings_w_protein = torch.cat([protein_embeds, embeddings], dim=1)
            encoder_outputs = self.model.encoder(
                inputs_embeds=embeddings_w_protein,
                attention_mask=protein_reactants_attention_mask,
                token_type_ids=token_type_ids,
            )

        else:
            encoder_attention_mask = encoder_input_ids["attention_mask"]
            # get reactant embeddings
            encoder_outputs = self.model.encoder(
                input_ids=encoder_input_ids["input_ids"],
                attention_mask=encoder_input_ids["attention_mask"],
                token_type_ids=token_type_ids,
            )
        return encoder_outputs, protein_reactants_attention_mask

    def forward(self, batch) -> Dict:

        if self.args.predict:
            predictions = self.generate(batch)
            return {
                "preds": predictions,
                "golds": batch["products"],
            }

        return_dict = self.config.use_return_dict

        # get molecule tokens
        encoder_input_ids = self.tokenize(batch["reactants"], self.tokenizer, self.args)
        decoder_input_ids = self.tokenize(batch["products"], self.tokenizer, self.args)

        # move to device
        for k, v in encoder_input_ids.items():
            encoder_input_ids[k] = v.to(self.devicevar.device)

        for k, v in decoder_input_ids.items():
            decoder_input_ids[k] = v.to(self.devicevar.device)

        # encode and get attention mask with proteins
        encoder_outputs, encoder_attention_mask = self.encode_reactants_and_enzyme(
            batch, encoder_input_ids
        )

        encoder_hidden_states = encoder_outputs[0]

        # attention mask with proteins
        # encoder_attention_mask = protein_reactants_attention_mask

        if self.append_enzyme_to_hiddens:
            encoder_hidden_states = torch.cat(
                [protein_embeds, encoder_hidden_states], dim=1
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
            input_ids=decoder_input_ids["input_ids"],
            attention_mask=decoder_input_ids["attention_mask"],
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=return_dict,
        )

        labels = decoder_input_ids["input_ids"].clone()
        labels[
            ~decoder_input_ids["attention_mask"].bool()
        ] = -100  # remove pad tokens from loss
        logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
        # transformers/models/bert/modeling_bert.py:1233
        # we are doing next-token prediction; shift prediction scores and input ids by one
        shifted_logits = logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shifted_logits.reshape(-1, self.model.decoder.config.vocab_size),
            labels.view(-1),
        )

        metric_labels = labels  # .view(-1)
        metric_logits = shifted_logits  # .reshape(-1, self.model.decoder.config.vocab_size)[metric_labels!= -100]
        metric_labels = metric_labels  # [ metric_labels!= -100]
        output = {
            "loss": loss,
            "logit": metric_logits,
            "y": metric_labels,
        }

        return output

    def generate(self, batch):
        """Applies auto-regressive generation for reaction prediction
        Usage: https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin

        Args:
            batch (dict): dictionary with input reactants
            decoder_start_token_id = 2,
            bos_token_id = 2,
            max_new_tokens=100
        """
        bos_id = self.tokenizer.cls_token_id
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id

        encoder_input_ids = self.tokenize(batch["reactants"], self.tokenizer, self.args)
        for k, v in encoder_input_ids.items():
            encoder_input_ids[k] = v.to(self.devicevar.device)

        encoder_outputs, encoder_attention_mask = self.encode_reactants_and_enzyme(
            batch, encoder_input_ids
        )

        generated_ids = self.model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            # input_ids=encoder_input_ids["input_ids"],
            decoder_start_token_id=bos_id,
            bos_token_id=bos_id,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            output_scores=True,
            return_dict_in_generate=True,
            **self.generation_config,
        )
        generated_samples = self.tokenizer.batch_decode(
            generated_ids.sequences, skip_special_tokens=True
        )
        generated_samples = [s.replace(" ", "") for s in generated_samples]
        beam_search_n = self.generation_config["num_return_sequences"]
        batch_samples = [
            generated_samples[i : i + beam_search_n]
            for i in range(0, len(generated_samples), beam_search_n)
        ]
        return batch_samples

    @staticmethod
    def add_args(parser) -> None:
        super(EnzymaticReactionEncoder, EnzymaticReactionEncoder).add_args(parser)
        parser.add_argument(
            "--enzyme_model",
            action=set_nox_type("model"),
            default=None,
            help="protein_model",
        )
        parser.add_argument(
            "--append_enzyme_to_hiddens",
            action="store_true",
            default=False,
            help="whether to append enzyme to hidden states. otherwise appended to inputs",
        )
        parser.add_argument(
            "--protein_representation_key",
            type=str,
            default="hidden",
            choices=["hidden", "token_hiddens"],
            help="which protein encoder output to uses",
        )
        parser.add_argument(
            "--protein_feature_dim",
            type=int,
            default=480,
            help="size of protein residue features from ESM models",
        )


@register_object("esm_decoder", "model")
class ESMDecoder(AbstractModel):
    def __init__(self, args):
        super().__init__()

        # make esm tokenizer + molecule tokenzier
        self.esm_tokenizer = AutoTokenizer.from_pretrained(
            args.esm_model_version
        )
        self.bert_tokenizer = self.init_tokenizer(args)
        esm_encoder_model = EsmModel.from_pretrained(args.esm_model_version)
        dconfig = self.get_transformer_config(args)
        dconfig.is_decoder = True
        dconfig.add_cross_attention = True
        bert_decoder_model = AutoModelForCausalLM.from_config(dconfig)
        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            encoder_model=esm_encoder_model, decoder_model=bert_decoder_model
        )
        self.config = self.model.config
        self.args = args
        self.register_buffer("devicevar", torch.zeros(1, dtype=torch.int8))

        self.generation_config = self.make_generation_config(args)

    def make_generation_config(self, args):
        generate_args = inspect.signature(self.model.generate).parameters
        args_dict = vars(args)
        gen_config = {}

        for k in generate_args:
            if f"generation_{k}" in args_dict:
                gen_config[k] = args_dict[f"generation_{k}"]

        return gen_config

    def get_transformer_config(self, args):
        if args.transformer_model == "bert":
            bert_config = BertConfig(
                max_position_embeddings=args.max_seq_len,
                vocab_size=self.bert_tokenizer.vocab_size,
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

        return bert_config

    @staticmethod
    def init_tokenizer(args):
        return BertTokenizer(
            vocab_file=args.vocab_path,
            do_lower_case=False,
            do_basic_tokenize=False,
            # sep_token=".",
            truncation=True,
            padding=True,
            model_max_length=args.max_seq_len,
            cls_token="[CLS]",
            eos_token="[EOS]",
        )

    @staticmethod
    def tokenize(list_of_text: List[str], tokenizer, args):
        # standardize reaction
        x = [standardize_reaction(r + ">>")[:-2] for r in list_of_text]
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

        if self.args.predict:
            predictions = self.generate(batch)
            return {
                "preds": predictions,
                "golds": batch["products"],
            }

        encoder_input_ids = self.esm_tokenizer(
            batch["sequence"],
            padding="longest",
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        decoder_input_ids = self.tokenize(
            batch["smiles"], self.bert_tokenizer, self.args
        )

        # move to device
        for k, v in encoder_input_ids.items():
            encoder_input_ids[k] = v.to(self.devicevar.device)

        for k, v in decoder_input_ids.items():
            decoder_input_ids[k] = v.to(self.devicevar.device)

        return_dict = self.config.use_return_dict

        encoder_outputs = self.model.encoder(
            input_ids=encoder_input_ids["input_ids"],
            attention_mask=encoder_input_ids["attention_mask"],
        )

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
            input_ids=decoder_input_ids["input_ids"],
            attention_mask=decoder_input_ids["attention_mask"],
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_input_ids["attention_mask"],
            return_dict=return_dict,
        )

        labels = decoder_input_ids["input_ids"].clone()
        labels[
            ~decoder_input_ids["attention_mask"].bool()
        ] = -100  # remove pad tokens from loss
        logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
        # transformers/models/bert/modeling_bert.py:1233
        # we are doing next-token prediction; shift prediction scores and input ids by one
        shifted_logits = logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shifted_logits.reshape(-1, self.model.decoder.config.vocab_size),
            labels.view(-1),
        )

        metric_labels = labels  # .view(-1)
        metric_logits = shifted_logits  # .reshape(-1, self.model.decoder.config.vocab_size)[metric_labels!= -100]
        metric_labels = metric_labels  # [ metric_labels!= -100]
        output = {
            "loss": loss,
            "logit": metric_logits,
            "y": metric_labels,
            # "decoder_hidden_states": decoder_outputs.hidden_states,
            # "past_key_values": decoder_outputs.past_key_values,
            # "decoder_hidden_states": decoder_outputs.hidden_states,
            # "decoder_attentions": decoder_outputs.attentions,
            # "cross_attentions": decoder_outputs.cross_attentions,
            # "encoder_last_hidden_state": encoder_outputs.last_hidden_state,
            # "encoder_hidden_states": encoder_outputs.hidden_states,
            # "encoder_attentions": encoder_outputs.attentions,
        }
        return output

    def generate(self, batch):
        """Applies auto-regressive generation for reaction prediction
        Usage: https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin

        Args:
            batch (dict): dictionary with input reactants
            decoder_start_token_id = 2,
            bos_token_id = 2,
            max_new_tokens=100
        """
        bos_id = self.tokenizer.cls_token_id
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id

        encoder_input_ids = self.tokenize(batch["reactants"], self.tokenizer, self.args)
        for k, v in encoder_input_ids.items():
            encoder_input_ids[k] = v.to(self.devicevar.device)

        generated_ids = self.model.generate(
            input_ids=encoder_input_ids["input_ids"],
            decoder_start_token_id=bos_id,
            bos_token_id=bos_id,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            attention_mask=encoder_input_ids["attention_mask"],
            output_scores=True,
            return_dict_in_generate=True,
            **self.generation_config,
        )
        generated_samples = self.tokenizer.batch_decode(
            generated_ids.sequences, skip_special_tokens=True
        )
        generated_samples = [s.replace(" ", "") for s in generated_samples]
        beam_search_n = self.generation_config["num_return_sequences"]
        batch_samples = [
            generated_samples[i : i + beam_search_n]
            for i in range(0, len(generated_samples), beam_search_n)
        ]
        return batch_samples

    @staticmethod
    def add_args(parser) -> None:
        parser.add_argument(
            "--esm_model_version",
            type=str,
            default="facebook/esm2_t33_650M_UR50D",
            help="which version of ESM to use"
        )
        parser.add_argument(
            "--transformer_model",
            type=str,
            default="bert",
            help="name of backbone model",
        )
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
        parser.add_argument(
            "--generation_max_new_tokens",
            type=int,
            default=200,
            help="The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.",
        )
        parser.add_argument(
            "--generation_num_beams",
            type=int,
            default=1,
            help="Number of beams for beam search. 1 means no beam search.",
        )
        parser.add_argument(
            "--generation_num_return_sequences",
            type=int,
            default=1,
            help="The number of independently computed returned sequences for each element in the batch. <= num_beams",
        )
        parser.add_argument(
            "--generation_num_beam_groups",
            type=int,
            default=1,
            help="Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams. [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details",
        )
        parser.add_argument(
            "--generation_do_sample",
            action="store_true",
            default=False,
            help="Whether or not to use sampling ; use greedy decoding otherwise.",
        )
        parser.add_argument(
            "--generation_early_stopping",
            action="store_true",
            default=False,
            help="Whether to stop the beam search when at least `num_beams` sentences are finished per batch or not",
        )
        parser.add_argument(
            "--generation_renormalize_logits",
            action="store_true",
            default=False,
            help=" Whether to renormalize the logits after applying all the logits processors or warpers (including the custom ones). It's highly recommended to set this flag to `True` as the search algorithms suppose the score logits are normalized but some logit processors or warpers break the normalization.",
        )