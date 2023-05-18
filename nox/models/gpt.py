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
    BertTokenizer,
    EsmModel,
    AutoTokenizer,
    GPTNeoConfig
)
from nox.models.modeling_gpt_neo import GPTNeoForCausalLM
import selfies as sf


@register_object("gpt_eznymatic_reaction_encoder", "model")
class GTPEnzymes(AbstractModel):
    def __init__(self, args):
        super().__init__()

        self.tokenizer = self.init_tokenizer(args)

        config = self.get_transformer_model(args)
        self.model = GPTNeoForCausalLM(config=config)
        self.config = self.model.config
        self.args = args
        self.register_buffer("devicevar", torch.zeros(1, dtype=torch.int8))

        self.generation_config = self.make_generation_config(args)

        self.use_selfies = getattr(args, 'use_selfies', False) 

        self.freeze_encoder = args.freeze_encoder
        
        if args.esm_model_version is not None:
            self.esm_tokenizer = AutoTokenizer.from_pretrained(args.esm_model_version)
            self.protein_model = EsmModel.from_pretrained(args.esm_model_version)
            if self.freeze_encoder:
                self.protein_model.eval()
        else:
            assert args.precomputed_esm_features_dir is not None

        self.protein_representation_key = args.protein_representation_key
        if args.hidden_size != args.protein_feature_dim:
            self.protein_fc = nn.Linear(args.protein_feature_dim, args.hidden_size)


    def make_generation_config(self, args):
        generate_args = inspect.signature(self.model.generate).parameters
        args_dict = vars(args)
        gen_config = {}

        for k in generate_args:
            if f"generation_{k}" in args_dict:
                gen_config[k] = args_dict[f"generation_{k}"]

        return gen_config

    def get_transformer_model(self, args):
        return GPTNeoConfig(
            vocab_size=self.tokenizer.vocab_size, 
            embedding_size=args.embedding_size,
            max_position_embeddings=args.max_seq_len, 
            hidden_size=args.hidden_size,
            num_layers=args.num_hidden_layers, 
            num_heads=args.num_heads, 
            intermediate_size=args.intermediate_size, 
            output_hidden_states=True,
            output_attentions=True,
            attention_types=[[['global', 'local'], args.num_hidden_layers//2 ]], 
            type_vocab_size=2,
            bos_token_id=self.tokenizer.bos_token_id, 
            eos_token_id=self.tokenizer.eos_token_id
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
            cls_token="[CLS]",
            eos_token="[EOS]",
        )

    @staticmethod
    def tokenize(list_of_text, tokenizer, args, prefix = "", suffix=""):

        # standardize reaction
        if getattr(args, "selfies", False):
            x = [ " ".join(sf.split_selfie(sf.encoder(r))) for r in list_of_text] 
        else:
            x = [standardize_reaction(r + ">>")[:-2] for r in list_of_text]
            x = [tokenize_smiles(r, return_as_str=True) for r in x]
        
        x = [f"{prefix} {r} {suffix}" for r in x]

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

    def encode_sequence(self, batch):
        if self.args.esm_model_version is not None:
            encoder_input_ids = self.esm_tokenizer(
                batch["sequence"],
                padding="longest",
                return_tensors="pt",
                return_special_tokens_mask=True,
            )
            for k, v in encoder_input_ids.items():
                encoder_input_ids[k] = v.to(self.devicevar.device)

            if self.freeze_encoder:
                with torch.no_grad():
                    encoder_outputs = self.protein_model(
                        input_ids=encoder_input_ids["input_ids"],
                        attention_mask=encoder_input_ids["attention_mask"],
                    )
            else:
                encoder_outputs = self.protein_model(
                    input_ids=encoder_input_ids["input_ids"],
                    attention_mask=encoder_input_ids["attention_mask"],
                )
            encoder_hidden_states = encoder_outputs
            return encoder_hidden_states, encoder_input_ids["attention_mask"]

    def encode_rxn_and_enzyme(self, batch, input_dict):
        # pass sequences through protein model or use precomputed hiddens
        if hasattr(self, "protein_model"):
            protein_output, protein_attention = self.encode_sequence(batch)
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

        # append [PROT] token
        protein_embeds = torch.cat([input_dict["protein_tok_embedding"].unsqueeze(1), protein_embeds], dim = 1)
        
        # prepend 1 in attention for prot token
        assert torch.all(protein_attention[:,[0]])
        protein_attention = torch.cat([protein_attention[:,[0]], protein_attention], dim = -1) 

        # combine protein - rxn attention mask
        protein_reactants_attention_mask = torch.cat(
                [protein_attention, input_dict["attention_mask"]], dim=-1
            )
        
        # token id types (protein vs reactants)
        token_type_ids = torch.cat(
            (torch.zeros_like(protein_attention), torch.ones_like(input_dict["attention_mask"])), dim = -1
        ).long()
        # torch.cat(
        #         [protein_attention * 0, input_dict["attention_mask"]], dim=-1
        #     ).long()

        # append embeddings
        embeddings_w_protein = torch.cat([protein_embeds, input_dict["inputs_embeds"]], dim=1)
        
        start_causal_index = protein_embeds.shape[1] + input_dict["reactant_max_len"]
        transformer_outputs = self.model.transformer(
            inputs_embeds=embeddings_w_protein,
            attention_mask=protein_reactants_attention_mask,
            # token_type_ids=token_type_ids,
            return_dict=True,
            output_attentions=True,
            output_hidden_states=True,
            start_causal_index=start_causal_index,
        )

        return transformer_outputs, protein_embeds

    def forward(self, batch) -> Dict:

        if self.args.predict:
            predictions = self.generate(batch)
            return {
                "preds": predictions["batch_samples"],
                "golds": batch["products"],
            }

        return_dict = self.config.use_return_dict

        # get molecule tokens
        reactant_input = self.tokenize(batch["reactants"], self.tokenizer, self.args, prefix="[RCT]", suffix="[PCT]")
        product_input = self.tokenize(batch["products"], self.tokenizer, self.args, suffix="[EOS]")

        input_ids = torch.cat([reactant_input["input_ids"], product_input["input_ids"]], dim = -1).to(self.devicevar.device)
        inputs_embeds = self.model.transformer.wte(input_ids)

        attention_mask = torch.cat([reactant_input["attention_mask"], product_input["attention_mask"]], dim = -1).to(self.devicevar.device)
        
        # get [PROT] token
        protein_tok_ids = torch.tensor( self.tokenizer.convert_tokens_to_ids(["[PROT]"]*len(batch["products"])) ).to(self.devicevar.device)
        protein_tok_embedding = self.model.transformer.wte(protein_tok_ids)

        # combine into reaction
        inputs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "reactant_max_len": reactant_input["input_ids"].shape[-1],
            "protein_tok_embedding": protein_tok_embedding,
        }

        # encode and get attention mask with proteins
        transformer_outputs, protein_embeds = self.encode_rxn_and_enzyme(batch, inputs)
        protein_seq_len = protein_embeds.shape[1]
        reactant_seq_len = reactant_input["input_ids"].shape[1]
        
        # so shape is length of product
        # substract 1 since [PCT] token is part of reactants
        hidden_states = transformer_outputs[0][:,(protein_seq_len+reactant_seq_len-1):] 

        lm_logits = self.model.lm_head(hidden_states)

        # loss function
        labels = product_input["input_ids"].clone()
        product_mask = product_input["attention_mask"]# True where product tokens exist
        labels[product_mask == 0] = -100  # remove pad tokens from loss 
        labels = labels.to(self.devicevar.device)
        
        # transformers/models/bert/modeling_bert.py:1233
        # we are doing next-token prediction; shift prediction scores and input ids by one
        shifted_logits = lm_logits[:, :-1, :].contiguous()
        #labels = labels[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shifted_logits.reshape(-1, self.model.config.vocab_size),
            labels.view(-1),
        )

        metric_logits = shifted_logits  # .reshape(-1, self.model.decoder.config.vocab_size)[metric_labels!= -100]

        output = {
            "loss": loss,
            "logit": metric_logits,
        }

        preds = torch.argmax(metric_logits.detach(), dim=-1)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=False)
        decoded_preds = [s.replace(" ", "") for s in decoded_preds]
        decoded_preds = [
            s[: s.index(self.tokenizer.eos_token)]
            if s.find(self.tokenizer.eos_token) > -1
            else s
            for s in decoded_preds
        ]  # stop at first eos
        output["pred_smiles"] = [s.split(".") for s in decoded_preds]

        # create predictions and labels to penalize errors due to early/late truncation but nothing after
        # get mask of 1s until first eos token for predictions
        pred_mask = (preds == self.tokenizer.eos_token_id).int().detach()
        pred_mask_has_eos = pred_mask.sum(-1)
        indices = torch.argmax(pred_mask, dim=1)
        pred_mask = torch.zeros_like(pred_mask)
        for r, i in enumerate(indices):
            if pred_mask_has_eos[r] > 0:
                pred_mask[r, : (i + 1)] = 1

        metric_labels = labels.clone()

        # get mask of 1s until first eos token that occurs last between prediction and label
        mask = ((metric_labels != -100) + pred_mask) > 0

        # prediction too long: make missing labels for extra (wrong) predictions different from predictions to count as wrong
        metric_labels[torch.logical_and(metric_labels == -100, mask)] = (
            preds[torch.logical_and(metric_labels == -100, mask)] + 1
        ) % self.args.num_classes
        # prediction too short: make missing predictions for labels different from labels to count as wrong
        preds[torch.logical_and(pred_mask == 0, mask)] = (
            metric_labels[torch.logical_and(pred_mask == 0, mask)] + 1
        ) % self.args.num_classes

        output["y"] = metric_labels
        output["preds"] = preds

        preds_mask = mask.int()
        preds_mask[preds_mask == 0] = -100
        output["preds_mask"] = preds_mask

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
        generated_raw_samples = self.tokenizer.batch_decode(
            generated_ids.sequences, skip_special_tokens=True
        )
        generated_samples = [s.replace(" ", "") for s in generated_raw_samples]
        beam_search_n = self.generation_config["num_return_sequences"]
        batch_samples = [
            generated_samples[i : i + beam_search_n]
            for i in range(0, len(generated_samples), beam_search_n)
        ]
        batch_raw_samples = [
            generated_raw_samples[i : i + beam_search_n]
            for i in range(0, len(generated_samples), beam_search_n)
        ]
        batch_ids = [
            generated_ids.sequences[i : i + beam_search_n]
            for i in range(0, len(generated_samples), beam_search_n)
        ]
        
        return {"batch_samples": batch_samples, "batch_raw_samples": batch_raw_samples, "batch_ids": batch_ids}

    
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
        parser.add_argument(
            "--use_selfies",
            action="store_true",
            default=False,
            help="use selfies instead of SMILES for reaction",
        )
        parser.add_argument(
            "--esm_model_version",
            type=str,
            default=None,
            help="which version of ESM to use",
        )
        parser.add_argument(
            "--protein_representation_key",
            type=str,
            default="hidden",
            choices=["hidden", "token_hiddens", "last_hidden_state", "pooler_output"],
            help="which protein encoder output to uses",
        )
        parser.add_argument(
            "--protein_feature_dim",
            type=int,
            default=480,
            help="size of protein residue features from ESM models",
        )

@register_object("gpt_corrector", "model")
class Corrector(GTPEnzymes):
    def __init__(self, args):
        super().__init__(args)
        state_dict = torch.load(args.generator_model_path)
        self.generator = EnzymaticReactionEncoder(state_dict['hyper_parameters']['args'])
        generator_state_dict = {k[len("model."):]: v for k,v in state_dict["state_dict"].items() }
        self.generator.load_state_dict(generator_state_dict)
        self.generator.eval()

        if args.corrector_model_path is not None:
            corrector_state_dict = torch.load(args.corrector_model_path)
            corrector_state_dict = {k[len("model."):]: v for k,v in corrector_state_dict["state_dict"].items() }
            self.model.load_state_dict({k[len("model."):]: v for k,v in corrector_state_dict.items() if k.startswith("model")})
            self.protein_model.load_state_dict({k[len("protein_model."):]: v for k,v in corrector_state_dict.items() if k.startswith("protein_model")})

        self.corrector_error_alpha = getattr(args, "corrector_error_alpha", 0.5)

    @staticmethod
    def add_args(parser) -> None:
        super(Corrector, Corrector).add_args(parser)
        parser.add_argument(
            "--generator_model_path",
            type=str,
            default=None,
            help="pretrained model to use",
        )
        parser.add_argument(
            "--corrector_model_path",
            type=str,
            default=None,
            help="pretrained model to use",
        )
        parser.add_argument(
            "--corrector_error_alpha",
            type=float,
            default=0.5,
            help="how much to weigh mistakes over correct generator predictions",
        )

    @staticmethod
    def tokenize(list_of_inputs: List[str], tokenizer, args, list_of_predictions = None):
        # standardize reaction
        
        if getattr(args, "selfies", False):
            x = [ " ".join(sf.split_selfie(sf.encoder(r))) for r in x] 
        else:
            x = [standardize_reaction(r + ">>")[:-2] for r in list_of_inputs]
            x = [tokenize_smiles(r, return_as_str=True) for r in x]

        if args.use_cls_token:
            # add [CLS] and [EOS] tokens
            x = [f"{tokenizer.cls_token} {r} {tokenizer.eos_token}" for r in x]

        if list_of_predictions is not None:
            list_of_predictions = [pred for sample_preds in list_of_predictions for pred in sample_preds] # flatten out
            x = [ "{} {} {}".format(given_input, tokenizer.sep_token, pred) for given_input, pred in zip(x, list_of_predictions)]


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

    def forward(self, batch):
        if self.args.predict:
            predictions = self.generate(batch)
            return {
                "preds": predictions,
                "golds": batch["products"],
            }

        return_dict = self.config.use_return_dict

        # generate product
        with torch.no_grad():
            # list of product(s)
            generation_dict = self.generator.generate(batch)
        
        generations = generation_dict["batch_raw_samples"]

        # add product to input
        reactants, products = [], []
        for idx, sample in enumerate(generations):
            reactants += [batch["reactants"][idx] for s in sample]
            products += [batch["products"][idx] for s in sample]

        # -------------------------------
        # same language modeling as above
        # -------------------------------

        # get molecule tokens
        encoder_input_ids = self.tokenize(reactants, self.tokenizer, self.args, generations)
        decoder_input_ids = self.tokenize(products, self.tokenizer, self.args)

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
        
        # weigh samples in loss according to generator accuracy
        gen_batch_ids = torch.cat(generation_dict['batch_ids'],dim=0) # generator predictions
        decoder_shape = min(gen_batch_ids.shape[-1], decoder_input_ids["input_ids"].shape[-1])
        weights = torch.eq(gen_batch_ids[:, :decoder_shape], decoder_input_ids["input_ids"][:,:decoder_shape]).all(1).int() # samples equal based on decoded ids
        weights = (1-weights)*self.corrector_error_alpha + weights*(1-self.corrector_error_alpha) # weight according to if correct (higher alpha -> weigh mistakes more)
        weights = weights.repeat_interleave(labels.shape[-1]) # number of tokens in loss function

        loss = torch.nn.functional.cross_entropy(
            shifted_logits.reshape(-1, self.model.decoder.config.vocab_size),
            labels.view(-1), 
            reduction="none") * weights
            
        loss = loss[torch.logical_and(labels.view(-1)!=-100, weights!=0)].mean()

        # loss = torch.nn.functional.cross_entropy(
        #     shifted_logits.reshape(-1, self.model.decoder.config.vocab_size),
        #     labels.view(-1),
        # )

        metric_logits = shifted_logits  # .reshape(-1, self.model.decoder.config.vocab_size)[metric_labels!= -100]

        output = {
            "loss": loss,
            "logit": metric_logits,
            "cross_attentions": decoder_outputs["cross_attentions"],
        }

        preds = torch.argmax(metric_logits.detach(), dim=-1)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=False)
        decoded_preds = [s.replace(" ", "") for s in decoded_preds]
        decoded_preds = [
            s[: s.index(self.tokenizer.eos_token)]
            if s.find(self.tokenizer.eos_token) > -1
            else s
            for s in decoded_preds
        ]  # stop at first eos
        output["pred_smiles"] = [s.split(".") for s in decoded_preds]

        # create predictions and labels to penalize errors due to early/late truncation but nothing after
        # get mask of 1s until first eos token for predictions
        pred_mask = (preds == self.tokenizer.eos_token_id).int().detach()
        pred_mask_has_eos = pred_mask.sum(-1)
        indices = torch.argmax(pred_mask, dim=1)
        pred_mask = torch.zeros_like(pred_mask)
        for r, i in enumerate(indices):
            if pred_mask_has_eos[r] > 0:
                pred_mask[r, : (i + 1)] = 1

        metric_labels = labels.clone()

        # get mask of 1s until first eos token that occurs last between prediction and label
        mask = ((metric_labels != -100) + pred_mask) > 0

        # prediction too long: make missing labels for extra (wrong) predictions different from predictions to count as wrong
        metric_labels[torch.logical_and(metric_labels == -100, mask)] = (
            preds[torch.logical_and(metric_labels == -100, mask)] + 1
        ) % self.args.num_classes
        # prediction too short: make missing predictions for labels different from labels to count as wrong
        preds[torch.logical_and(pred_mask == 0, mask)] = (
            metric_labels[torch.logical_and(pred_mask == 0, mask)] + 1
        ) % self.args.num_classes

        output["y"] = metric_labels
        output["preds"] = preds

        preds_mask = mask.int()
        preds_mask[preds_mask == 0] = -100
        output["preds_mask"] = preds_mask

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