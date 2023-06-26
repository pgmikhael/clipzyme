import torch 
import torch.nn as nn
import torch.nn.functional as F
from nox.utils.registry import get_object, register_object
from nox.utils.classes import set_nox_type
from nox.utils.pyg import unbatch
from nox.utils.wln_processing import generate_candidates_from_scores, get_batch_candidate_bonds, robust_edit_mol
from nox.models.abstract import AbstractModel
from torch_scatter import scatter, scatter_add
from torch_geometric.utils import to_dense_batch, to_dense_adj
from collections import defaultdict
from nox.models.gat import GAT
from nox.models.chemprop import DMPNNEncoder
from rdkit import Chem 
import copy 
import os 

class WLDN_Cache:
    def __init__(self, path, extension="pt"):
        if not os.path.exists(path):
            os.makedirs(path)
        self.cache_dir = path
        self.files_extension = extension

    def _file_path(self, sample_id):
        return os.path.join(self.cache_dir, f"{sample_id}_candidates.pt")

    def exists(self, sample_id):
        return os.path.isfile(self._file_path(sample_id))

    def get(self, sample_id):
        return torch.load(self._file_path(sample_id))

    def add(self, sample_id, graph):
        torch.save(graph, self._file_path(sample_id))


@register_object("cheap_global_attn", "model")
class CheapGlobalAttention(AbstractModel):
    def __init__(self, args):
        super(CheapGlobalAttention, self).__init__()
        self.linear = nn.Linear(args.gat_hidden_dim, 1)
        
    def forward(self, node_feats, batch_index):
        # node_feats is (N, in_dim)
        scores = self.linear(node_feats)  # (N, 1)
        scores = torch.softmax(scores, dim=0)  # softmax over all nodes
        scores = scores.squeeze(1)  # (N, )
        out = scatter_add(node_feats * scores.unsqueeze(-1), batch_index, dim=0)
        return out

@register_object("pairwise_global_attn", "model")
class PairwiseAttention(AbstractModel):
    def __init__(self, args):
        super(PairwiseAttention, self).__init__()
        self.P_a = nn.Linear(args.gat_hidden_dim, args.gat_hidden_dim)
        self.P_b = nn.Linear(args.gat_complete_edge_dim, args.gat_hidden_dim)
        self.U = nn.Linear(args.gat_hidden_dim, 1)
        

    def forward(self, node_feats, graph):
        """Compute node contexts with global attention
        
        node_feats: N x F, where N is number of nodes, F is feature dimension
        graph: batched graph with edges of complete graph
        """
        # Batch index: N, mapping each node to corresponding graph index
        edge_index_complete = graph.edge_index_complete
        edge_attr_complete = graph.edge_attr_complete
        batch_index = graph.batch

        # Project features
        node_feats_transformed = self.P_a(node_feats)  # N x F
        edge_feats_complete = self.P_b(edge_attr_complete.float()) # E x F

         # convert to dense adj: E x F -> N x N x F
        dense_edge_attr = to_dense_adj(edge_index = edge_index_complete, edge_attr = edge_feats_complete).squeeze(0)

        # node_features: sparse batch: N x D
        pairwise_node_feats = node_feats_transformed.unsqueeze(1) + node_feats_transformed # N x N x F
        
        # Compute attention scores
        scores = torch.sigmoid(self.U(F.relu(pairwise_node_feats + dense_edge_attr))).squeeze(-1)

        # Mask attention scores to prevent attention to nodes in different graphs
        mask = batch_index[:, None] != batch_index[None, :]  # N x N
        weights = scores.masked_fill(mask, 0)

        # Apply attention weights
        weighted_feats = torch.matmul(weights, node_feats)  # N x F

        return weighted_feats # node_contexts


@register_object("gatv2_globalattn", "model")
class GATWithGlobalAttn(GAT):
    def __init__(self, args):
        super().__init__(args)
        self.global_attention = get_object(args.attn_type, "model")(args)

    def forward(self, graph):
        output = super().forward(graph) # Graph NN (GAT) Local Network

        weighted_node_feats = self.global_attention(output["node_features"], graph)  # EQN 6

        output["node_features_attn"] = weighted_node_feats
        return output

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        super(GATWithGlobalAttn, GATWithGlobalAttn).add_args(parser)
        parser.add_argument(
            "--attn_type",
            type=str,
            action=set_nox_type("model"),
            default="pairwise_global_attn",
            help="type of global attention to use"
        )


@register_object("reaction_center_net", "model")
class ReactivityCenterNet(AbstractModel):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.gat_global_attention = get_object(args.gat_type, "model")(args) # GATWithGlobalAttn(args)
        self.M_a = nn.Linear(args.gat_hidden_dim, args.gat_hidden_dim)
        self.M_b = nn.Linear(args.gat_complete_edge_dim, args.gat_hidden_dim)
        self.lin = nn.Linear(2*args.gat_hidden_dim, args.gat_hidden_dim)
        self.U = nn.Sequential(
            nn.ReLU(),
            nn.Linear(args.gat_hidden_dim, args.num_predicted_bond_types) 
        )

    def forward(self, batch):
        gat_output = self.gat_global_attention(batch['reactants']) # GAT + Global Attention over node features
        cs = gat_output["node_features"]
        c_tildes = gat_output["node_features_attn"] # node contexts
        c_final = self.lin(torch.cat([cs, c_tildes], dim=-1)) # N x 2*hidden_dim -> N x hidden_dim

        s_uv = self.forward_helper(c_final, batch['reactants']['edge_index_complete'], batch['reactants']['edge_attr_complete'], batch['reactants']['batch'])

        # precompute for top k metric
        candidate_bond_changes = get_batch_candidate_bonds(batch["reaction"], s_uv.detach(), batch['reactants'].batch)
        # make bonds that are "4" -> "1.5"
        for i in range(len(candidate_bond_changes)):
            candidate_bond_changes[i] = [(elem[0], elem[1], 1.5, elem[3]) if elem[2] == 4 else elem for elem in candidate_bond_changes[i]]

        batch_real_bond_changes = []
        for i in range(len(batch['reactants']['bond_changes'])):
            reaction_real_bond_changes = []
            for elem in batch['reactants']['bond_changes'][i]:
                reaction_real_bond_changes.append(tuple(elem))
            batch_real_bond_changes.append(reaction_real_bond_changes)

        assert len(candidate_bond_changes) == len(batch_real_bond_changes)

        return {
            "s_uv": s_uv,
            "candidate_bond_changes": candidate_bond_changes,
            "real_bond_changes": batch_real_bond_changes
        }

    def forward_helper(self, node_features, edge_indices, edge_attr, batch_indices):
        # GAT with global attention
        node_features = self.M_a(node_features) # N x hidden_dim -> N x hidden_dim 
        edge_attr = self.M_b(edge_attr.float()) # E x 5 -> E x hidden_dim 

        # convert to dense adj: E x hidden_dim -> N x N x hidden_dim
        dense_edge_attr = to_dense_adj(edge_index = edge_indices, edge_attr = edge_attr).squeeze(0)

        # node_features: sparse batch: N x D
        pairwise_node_feats = node_features.unsqueeze(1) + node_features # N x N x D
        # edge_attr: bond features: N x N x D
        s = self.U(dense_edge_attr + pairwise_node_feats).squeeze(-1) # N x N
        # removed this line since the sizes become inconsistent later
        # s, mask = to_dense_batch(s, batch_indices) # B x max_batch_N x N x num_predicted_bond_types
        
        # make symmetric
        s = (s + s.transpose(0,1))/2
        return s

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        super(ReactivityCenterNet, ReactivityCenterNet).add_args(parser)
        parser.add_argument(
            "--num_predicted_bond_types",
            type=int,
            default=5,
            help="number of bond types to predict, this is t in the paper"
        )
        parser.add_argument(
            "--gat_type",
            type=str,
            action=set_nox_type("model"),
            default="gatv2_globalattn",
            help="Type of gat to use, mainly to init args"
        )
        parser.add_argument(
            "--topk_bonds",
            nargs='+',
            type=int,
            default=[1, 3, 5],
            help="topk bonds to consider for accuracy metric"
        )
        parser.add_argument(
            "--gat_complete_edge_dim",
            type=int,
            default=5,
            help="dimension of edges in complete graph"
        )


@register_object("wldn", "model")
class WLDN(AbstractModel):
    def __init__(self, args):
        super().__init__()
        self.args = args
        try:
            state_dict = torch.load(args.reactivity_model_path)
            self.reactivity_net = get_object(args.reactivity_net_type, "model")(state_dict['hyper_parameters']['args'])
            self.reactivity_net.load_state_dict({k[len("model."):]: v for k,v in state_dict["state_dict"].items() if k.startswith("model")})
            self.reactivity_net.requires_grad_(False)
        except:
            self.reactivity_net = get_object(args.reactivity_net_type, "model")(args).requires_grad_(False)
            print("Could not load pretrained model")
        self.reactivity_net.eval()
        
        wln_diff_args = copy.deepcopy(args)
        # GAT
        self.wln = GAT(args) # WLN for mol representation GAT(args)
        wln_diff_args = copy.deepcopy(args)
        wln_diff_args.gat_node_dim  = args.gat_hidden_dim
        wln_diff_args.gat_num_layers = 1
        self.wln_diff = GAT(wln_diff_args)
        self.final_transform = nn.Sequential(
            nn.Linear(args.gat_hidden_dim, args.gat_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.gat_hidden_dim, 1)
        )
        self.use_cache = args.cache_path is not None 
        if self.use_cache:
            self.cache = WLDN_Cache(os.path.join(args.cache_path), "pt")
        if self.args.predict:
            assert not self.args.train 

    def predict(self, batch, product_candidates_list, candidate_scores):
        predictions = []
        for idx, (candidates, scores) in enumerate(zip(product_candidates_list, candidate_scores)):
            smiles_predictions = []
            # sort according to ranker score
            scores_indices = torch.argsort(scores.view(-1),descending=True)
            valid_candidate_combos = [candidates.candidate_bond_change[i] for i in scores_indices]
            reactant_mol = Chem.MolFromSmiles(batch["smiles"][idx])
            for edits in valid_candidate_combos:
                smiles = robust_edit_mol(reactant_mol, edits)
                if len(smiles) != 0:
                    smiles_predictions.append(smiles)
                try:
                    Chem.Kekulize(reactant_mol)
                    smiles = robust_edit_mol(reactant_mol, edits)
                    smiles_predictions.append(smiles)
                except Exception as e:
                    smiles_predictions.append(smiles)
            predictions.append(smiles_predictions)
            
        return {"preds": predictions}

    def forward(self, batch):
        product_candidates_list = self.get_product_candidate_list(batch, batch["sample_id"])

        reactant_node_feats = self.wln(batch["reactants"])["node_features"] # N x D, where N is all the nodes in the batch
        dense_reactant_node_feats, mask = to_dense_batch(reactant_node_feats, batch=batch["reactants"].batch) # B x max_batch_N x D
        candidate_scores = []
        for idx, product_candidates in enumerate(product_candidates_list):
            # get node features for candidate products
            product_candidates = product_candidates.to(reactant_node_feats.device)
            candidate_node_feats = self.wln(product_candidates)["node_features"]
            dense_candidate_node_feats, mask = to_dense_batch(candidate_node_feats, batch=product_candidates.batch) # B x num_nodes x D
            
            num_nodes = dense_candidate_node_feats.shape[1]

            # compute difference vectors and replace the node features of the product graph with them
            difference_vectors = dense_candidate_node_feats - dense_reactant_node_feats[idx][:num_nodes].unsqueeze(0)

            # undensify
            total_nodes = dense_candidate_node_feats.shape[0] * num_nodes
            difference_vectors = difference_vectors.view(total_nodes, -1)
            product_candidates.x = difference_vectors
            
            # apply a separate WLN to the difference graph
            wln_diff_output = self.wln_diff(product_candidates)
            diff_node_feats = wln_diff_output["node_features"]

            # compute the score for each candidate product
            # to dense
            diff_node_feats, _ = to_dense_batch(diff_node_feats, product_candidates.batch) # num_candidates x max_num_nodes x D
            score = self.final_transform(torch.sum(diff_node_feats, dim=-2))
            if self.args.add_core_score:
                core_scores = [sum(c[-1] for c in cand_changes) for cand_changes in product_candidates.candidate_bond_change]
                score = score + torch.tensor(core_scores, device = score.device).unsqueeze(-1)
            candidate_scores.append(score) # K x 1

        # ! PREDICT SMILES
        if self.args.predict:
            return self.predict(batch, product_candidates_list, candidate_scores)

        # note: dgl implementation adds reactivity score
        output = {
            "logit": candidate_scores,
            "product_candidates_list": product_candidates_list,
            # "s_uv": reactivity_output["s_uv"], # for debugging purposes
            }
        return output

    # seperate function because stays the same for different forward methods
    def get_product_candidate_list(self, batch, sample_ids):
        """
        Args:
            batch : collated samples from dataloader
            sample_ids: list of sample ids
        """
        mode = "train" # this is used to get candidates, using robust_edit_mol when in predict later for actual smiles generation

        if self.use_cache:
            if not all( self.cache.exists(sid) for sid in sample_ids ):
                with torch.no_grad():
                    reactivity_output = self.reactivity_net(batch) # s_uv: N x N x 5, 'candidate_bond_changes', 'real_bond_changes'

                # get candidate products as graph structures
                # each element in this list is a batch of candidate products (where each batch represents one reaction)
                product_candidates_list = generate_candidates_from_scores(reactivity_output, batch, self.args, mode)
                [self.cache.add(sid, product_candidates) for sid, product_candidates in zip(sample_ids, product_candidates_list)]
            else:
                product_candidates_list =  [self.cache.get(sid) for sid in sample_ids]
        else:
            # each element in this list is a batch of candidate products (where each batch represents one reaction)
            with torch.no_grad():
                reactivity_output = self.reactivity_net(batch) # s_uv: N x N x 5, 'candidate_bond_changes', 'real_bond_changes'

            product_candidates_list = generate_candidates_from_scores(reactivity_output, batch, self.args, mode)

        return product_candidates_list
        

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        super(WLDN, WLDN).add_args(parser)
        parser.add_argument(
            "--num_candidate_bond_changes",
            type=int,
            default=20,
            help="Core size"
        )
        parser.add_argument(
            "--max_num_bond_changes",
            type=int,
            default=5,
            help="Combinations"
        )
        parser.add_argument(
            "--max_num_change_combos_per_reaction",
            type=int,
            default=500,
            help="cutoff"
        )
        parser.add_argument(
            "--reactivity_net_type",
            type=str,
            action=set_nox_type("model"),
            default="reaction_center_net",
            help="Type of reactivity net to use, mainly to init args"
        )
        parser.add_argument(
            "--reactivity_model_path",
            type=str,
            help="path to pretrained reaction center prediction model"
        )
        parser.add_argument(
            "--add_core_score",
            action="store_true",
            default=False,
            help="whether to add core score to ranking prediction"
        )

##########################################################################################

##########################################################################################

##########################################################################################

##########################################################################################

##########################################################################################

from transformers import BertConfig, BertModel, AutoTokenizer, EsmModel
@register_object("wldn_transformer", "model")
class WLDNTransformer(WLDN):
    def __init__(self, args):
        super().__init__(args) # gives self.wln (GAT)
        self.esm_tokenizer = AutoTokenizer.from_pretrained(args.esm_model_version)
        self.esm_model = EsmModel.from_pretrained(args.esm_model_version)
        self.freeze_encoder = args.freeze_encoder
        if self.freeze_encoder:
            self.esm_model.eval()
        econfig = self.get_transformer_config(args)
        econfig.is_decoder = False
        self.model = BertModel(econfig, add_pooling_layer=True)
        self.config = self.model.config
        self.args = args
        self.register_buffer("token_type_ids", torch.zeros(1, dtype=torch.long))

        self.final_transform = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, 1)
        )  # need to overwrite because output is different size

    def get_transformer_config(self, args):
        bert_config = BertConfig(
                # vocab_size=self.bert_tokenizer.vocab_size,
                max_position_embeddings=args.max_seq_len,
                type_vocab_size = 2,
                output_hidden_states=True,
                output_attentions=True,
                hidden_size=args.hidden_size,
                intermediate_size=args.intermediate_size,
                num_attention_heads=args.num_heads,
                num_hidden_layers=args.num_hidden_layers,
            )
        return bert_config

    def encode_sequence(self, batch):
        encoder_input_ids = self.esm_tokenizer(
            batch["sequence"],
            padding="longest",
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        for k, v in encoder_input_ids.items():
            encoder_input_ids[k] = v.to(self.token_type_ids.device)

        if self.freeze_encoder:
            self.esm_model.requires_grad_(False)
            with torch.no_grad():
                encoder_outputs = self.esm_model(
                    input_ids=encoder_input_ids["input_ids"],
                    attention_mask=encoder_input_ids["attention_mask"],
                )
        else:
            encoder_outputs = self.esm_model(
                input_ids=encoder_input_ids["input_ids"],
                attention_mask=encoder_input_ids["attention_mask"],
            )
        encoder_hidden_states = encoder_outputs["last_hidden_state"]
        return encoder_hidden_states, encoder_input_ids["attention_mask"]

    def forward(self, batch):
        prot_feats, prot_attn = self.encode_sequence(batch) # 1 x len_seq x hidden_dim

        product_candidates_list = self.get_product_candidate_list(batch, batch["row_id"])

        reactant_node_feats = self.wln(batch["reactants"])["node_features"] # N x D, where N is all the nodes in the batch
        dense_reactant_node_feats, mask = to_dense_batch(reactant_node_feats, batch=batch["reactants"].batch) # B x max_batch_N x D
        candidate_scores = []
        for idx, product_candidates in enumerate(product_candidates_list):
            # get node features for candidate products
            product_candidates = product_candidates.to(reactant_node_feats.device)
            candidate_node_feats = self.wln(product_candidates)["node_features"]
            dense_candidate_node_feats, candidate_mask = to_dense_batch(candidate_node_feats, batch=product_candidates.batch) # B x num_nodes x D
            
            num_nodes = dense_candidate_node_feats.shape[1]

            # compute difference vectors and replace the node features of the product graph with them
            diff_node_feats = dense_candidate_node_feats - dense_reactant_node_feats[idx][:num_nodes].unsqueeze(0)

            num_candidates = diff_node_feats.shape[0]
            # repeat protein features and mask to shape
            repeated_prot = prot_feats[idx].unsqueeze(0).repeat(num_candidates, 1, 1)
            prot_mask = prot_attn[idx].unsqueeze(0).repeat(num_candidates, 1)
            
            # concatenate the prot features with the product features
            concatenated_feats = torch.cat([diff_node_feats, repeated_prot], dim=1) # num_candidates x (max_batch_N + len_seq) x D

            # create token_type_ids tensor
            token_type_ids = self.token_type_ids.repeat(*concatenated_feats.shape[:2])  # Initialize with zeros
            token_type_ids[:, num_nodes:] = 1  # Assign token type 1 to the second sequence

            # compute the attention mask so that padded product features are not attended to
            attention_mask = torch.cat([candidate_mask, prot_mask], dim=1) # num_candidates x (max_batch_N + len_seq)

            outputs = self.model(
                inputs_embeds= concatenated_feats,
                attention_mask= attention_mask,
                token_type_ids= token_type_ids
            )

            score = self.final_transform(outputs["pooler_output"])
            if self.args.add_core_score:
                core_scores = [sum(c[-1] for c in cand_changes) for cand_changes in product_candidates.candidate_bond_change]
                score = score + torch.tensor(core_scores, device = score.device).unsqueeze(-1)
            candidate_scores.append(score) # K x 1
        
        output = {
            "logit": candidate_scores,
            "product_candidates_list": product_candidates_list,
            }
        return output

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        super(WLDNTransformer, WLDNTransformer).add_args(parser)
        parser.add_argument(
            "--esm_model_version",
            type=str,
            default=None,
            help="which version of ESM to use",
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
            "--num_heads",
            type=int,
            default=8,
            help="maximum length allowed for the input sequence",
        )
        


###########################################################################################

# class EnzymeMoleculeBERT(AbstractModel):
#     def __init__(self, args):
#         super().__init__()
#         self.esm_tokenizer = AutoTokenizer.from_pretrained(args.esm_model_version)
#         self.esm_model = EsmModel.from_pretrained(args.esm_model_version)
#         if self.freeze_encoder:
#             self.esm_model.eval()
#         econfig = self.get_transformer_config(args)
#         econfig.is_decoder = False
#         self.model = BertModel(econfig, add_pooling_layer=False)
#         self.config = self.model.config
#         self.args = args
#         self.register_buffer("devicevar", torch.zeros(1, dtype=torch.int8))
#         self.generation_config = self.make_generation_config(args)

#     def get_transformer_config(self, args):
#         if args.transformer_model == "bert":
#             bert_config = BertConfig(
#                 # max_position_embeddings=args.max_seq_len,
#                 # vocab_size=3,
#                 output_hidden_states=True,
#                 output_attentions=True,
#                 hidden_size=args.hidden_size,
#                 intermediate_size=args.intermediate_size,
#                 embedding_size=args.embedding_size,
#                 num_attention_heads=args.num_heads,
#                 num_hidden_layers=args.num_hidden_layers,
#             )
#         else:
#             raise NotImplementedError

#         return bert_config

#     def forward(self, batch) -> Dict:

#         forward_args = {
#             "input_ids": None,
#             # If padding apply attention mask on padding (below)
#             "attention_mask": # TODO,
#             # torch.LongTensor of shape (batch_size, sequence_length)
#             "token_type_ids": # TODO, indicates which is prot and which is molecule,
#             "position_ids": None,
#             "head_mask": None,
#             # encoder inputs can be (batch, sec_len, hidden_dim) so need to pad to max_seq_len
#             "inputs_embeds": # TODO put input here,
#             "encoder_hidden_states": None,
#             "encoder_attention_mask": None,
#             "past_key_values": None,
#             "use_cache": None,
#             "output_attentions": False,
#             "output_hidden_states": True,
#             "return_dict": True,
#         }

#         logits = decoder_outputs.logits if return_dict else decoder_outputs[0]

#         output = {
#             "logit": logits,
#         }

        
#         return output

#     @staticmethod
#     def add_args(parser) -> None:
#         parser.add_argument(
#             "--esm_model_version",
#             type=str,
#             default="facebook/esm2_t33_650M_UR50D",
#             help="which version of ESM to use",
#         )
#         parser.add_argument(
#             "--transformer_model",
#             type=str,
#             default="bert",
#             help="name of backbone model",
#         )
#         parser.add_argument(
#             "--vocab_path",
#             type=str,
#             default=None,
#             required=True,
#             help="path to vocab text file required for tokenizer",
#         )
#         parser.add_argument(
#             "--merges_file_path",
#             type=str,
#             default=None,
#             help="path to merges file required for RoBerta tokenizer",
#         )
#         parser.add_argument(
#             "--freeze_encoder",
#             action="store_true",
#             default=False,
#             help="whether use model as pre-trained encoder and not update weights",
#         )
#         parser.add_argument(
#             "--num_hidden_layers",
#             type=int,
#             default=6,
#             help="number of layers in the transformer",
#         )
#         parser.add_argument(
#             "--max_seq_len",
#             type=int,
#             default=512,
#             help="maximum length allowed for the input sequence",
#         )
#         parser.add_argument(
#             "--hidden_size",
#             type=int,
#             default=256,
#             help="maximum length allowed for the input sequence",
#         )
#         parser.add_argument(
#             "--intermediate_size",
#             type=int,
#             default=512,
#             help="maximum length allowed for the input sequence",
#         )
#         parser.add_argument(
#             "--embedding_size",
#             type=int,
#             default=128,
#             help="maximum length allowed for the input sequence",
#         )
#         parser.add_argument(
#             "--num_heads",
#             type=int,
#             default=8,
#             help="maximum length allowed for the input sequence",
#         )
#         parser.add_argument(
#             "--do_masked_language_model",
#             "-mlm",
#             action="store_true",
#             default=False,
#             help="whether to perform masked language model task",
#         )
#         parser.add_argument(
#             "--mlm_probability",
#             type=float,
#             default=0.1,
#             help="probability that a token chosen to be masked. IF chosen, 80% will be masked, 10% random, 10% original",
#         )
#         parser.add_argument(
#             "--use_cls_token",
#             action="store_true",
#             default=False,
#             help="use cls token as hidden representation of sequence",
#         )
#         parser.add_argument(
#             "--hidden_aggregate_func",
#             type=str,
#             default="mean",
#             choices=["mean", "sum", "cls"],
#             help="use cls token as hidden representation of sequence",
#         )
#         parser.add_argument(
#             "--longformer_model_path",
#             type=str,
#             default=None,
#             help="path to saved model if loading from pretrained",
#         )
#         parser.add_argument(
#             "--generation_max_new_tokens",
#             type=int,
#             default=200,
#             help="The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.",
#         )
#         parser.add_argument(
#             "--generation_num_beams",
#             type=int,
#             default=1,
#             help="Number of beams for beam search. 1 means no beam search.",
#         )
#         parser.add_argument(
#             "--generation_num_return_sequences",
#             type=int,
#             default=1,
#             help="The number of independently computed returned sequences for each element in the batch. <= num_beams",
#         )
#         parser.add_argument(
#             "--generation_num_beam_groups",
#             type=int,
#             default=1,
#             help="Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams. [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details",
#         )
#         parser.add_argument(
#             "--generation_do_sample",
#             action="store_true",
#             default=False,
#             help="Whether or not to use sampling ; use greedy decoding otherwise.",
#         )
#         parser.add_argument(
#             "--generation_early_stopping",
#             action="store_true",
#             default=False,
#             help="Whether to stop the beam search when at least `num_beams` sentences are finished per batch or not",
#         )
#         parser.add_argument(
#             "--generation_renormalize_logits",
#             action="store_true",
#             default=False,
#             help=" Whether to renormalize the logits after applying all the logits processors or warpers (including the custom ones). It's highly recommended to set this flag to `True` as the search algorithms suppose the score logits are normalized but some logit processors or warpers break the normalization.",
#         )
