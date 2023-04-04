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
from torch_geometric.utils import to_dense_batch, to_dense_adj

@register_object("esm_class_decoder", "model")
class ESMClassDecoder(AbstractModel):
    def __init__(self, args):
        super().__init__()

        # make esm tokenizer + molecule tokenzier
        self.esm_tokenizer = AutoTokenizer.from_pretrained(
            args.esm_model_version
        )
        
        esm_encoder_model = EsmModel.from_pretrained(args.esm_model_version)
        dconfig = self.get_transformer_config(args)
        dconfig.is_decoder = True
        dconfig.add_cross_attention = True
        bert_decoder_model = AutoModel.from_config(dconfig) #
        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            encoder_model=esm_encoder_model, decoder_model=bert_decoder_model
        )
        self.classifier = nn.Linear(args.hidden_size, 1)
        self.config = self.model.config
        self.args = args
        self.register_buffer("devicevar", torch.zeros(1, dtype=torch.int8))
        self.register_buffer("class_idx", torch.arange(args.num_classes))

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
                max_position_embeddings=args.num_classes,
                vocab_size=args.num_classes,
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


    def forward(self, batch) -> Dict:
        batch['sequence'] = ["NRGAMG", "NRGAMGT"]

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
        batch_size = len(batch["sequence"])
        
        decoder_input_ids = self.class_idx.unsqueeze(0).repeat_interleave(batch_size,0)
        
        # move to device
        for k, v in encoder_input_ids.items():
            encoder_input_ids[k] = v.to(self.devicevar.device)

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
        decoder_attention = torch.ones_like(decoder_input_ids).unsqueeze(-1).repeat_interleave(decoder_input_ids.shape[-1],-1) # n x n so that it's not replaced by LM
        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_input_ids["attention_mask"],
            return_dict=return_dict,
        )

        logits = self.classifier(decoder_outputs[0]).view(batch_size, -1)
        output = {
            "logit": logits,
        }
        return output

    def generate(self, batch):
        '''Applies auto-regressive generation for reaction prediction
        Usage: https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin

        Args:
            batch (dict): dictionary with input reactants
            decoder_start_token_id = 2,
            bos_token_id = 2,
            max_new_tokens=100
        '''
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

@register_object("esm_graph_decoder", "model")
class ESMGraphDecoder(AbstractModel):
    def __init__(self, args):
        super().__init__()

        # make esm tokenizer + molecule tokenzier
        self.esm_tokenizer = AutoTokenizer.from_pretrained(
            args.esm_model_version
        )
        
        esm_encoder_model = EsmModel.from_pretrained(args.esm_model_version)
        dconfig = self.get_transformer_config(args)
        dconfig.is_decoder = True
        dconfig.add_cross_attention = True
        bert_decoder_model = AutoModel.from_config(dconfig)
        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            encoder_model=esm_encoder_model, decoder_model=bert_decoder_model
        )
        self.graph_encoder = get_object(args.graph_encoder, "model")(args)
        self.config = self.model.config
        self.args = args
        self.register_buffer("devicevar", torch.zeros(1, dtype=torch.int8))
        self.freeze_encoder = args.freeze_encoder

    def get_transformer_config(self, args):
        if args.transformer_model == "bert":
            bert_config = BertConfig(
                max_position_embeddings=args.max_seq_len,
                vocab_size=1,
                output_hidden_states=True,
                output_attentions=True,
                hidden_size=args.hidden_size,
                intermediate_size=args.intermediate_size,
                embedding_size=1,
                num_attention_heads=args.num_heads,
                num_hidden_layers=args.num_hidden_layers,
            )
        else:
            raise NotImplementedError

        return bert_config

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
        

        # move to device
        for k, v in encoder_input_ids.items():
            encoder_input_ids[k] = v.to(self.devicevar.device)

        return_dict = self.config.use_return_dict

        if self.freeze_encoder:
            with torch.no_grad():
                encoder_outputs = self.model.encoder(
                    input_ids=encoder_input_ids["input_ids"],
                    attention_mask=encoder_input_ids["attention_mask"],
                )
        else:
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
        

        # get graph encoding (sparse)
        decoder_inputs = self.graph_encoder(batch["smiles"])
        # make into dense
        decoder_inputs, decoder_mask = to_dense_batch(
            x=decoder_inputs["node_features"], 
            batch=batch["smiles"].batch 
            )

        # to_dense adjacency
        # make 2d attention so that it's not replaced by causal LM
        decoder_attention = to_dense_adj(
            edge_index=batch["smiles"].edge_index,
            batch=batch["smiles"].batch, 
            # edge_attr=batch["smiles"].edge_attr,
        )

        # add CLS token to graph input
        B, N, d = decoder_inputs.shape 
        cls_token = self.model.decoder.embeddings(torch.zeros((B,1), dtype=torch.long, device=self.devicevar.device))
        decoder_inputs = torch.concat([cls_token, decoder_inputs], dim = 1) # B, N+1, d
        decoder_mask = torch.concat([decoder_mask[:,0].unsqueeze(-1), decoder_mask], dim = 1) # add column of ones for CLS
        
        cls_token_attn = torch.ones((B,N+1,N+1), device=self.devicevar.device)
        # cls_token_attn[:,1:,1:] = decoder_attention

        # Decode
        decoder_outputs = self.model.decoder(
            inputs_embeds=decoder_inputs,
            attention_mask=cls_token_attn,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_input_ids["attention_mask"],
            return_dict=return_dict,
        )

        output = {
            "hidden": decoder_outputs[0][:,0] # cls token
        }
        return output

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
            "--graph_encoder",
            action=set_nox_type("model"),
            default="chemprop",
            help="name graph encoder to use",
        )

