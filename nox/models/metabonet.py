import torch.nn as nn
import torch.nn.functional as F
import torch
from nox.utils.registry import register_object
from nox.models.abstract import AbstractModel
from nox.models.classifier import MLPClassifier, GraphClassifier
from torch_geometric.nn.conv.gatv2_conv import GAT
from torch_scatter import scatter


@register_object("metabonet_pathways")
class MetaboNetPathways(AbstractModel):
    def __init__(self, args):
        super(MetaboNetPathways, self).__init__()

        self.args = args
        self.molecule_encoder = GraphClassifier(args)
        self.gsm_encoder = GAT(args)
        self.mlp = MLPClassifier(args)

    def forward(self, batch):
        # Encode the molecules
        molecule_embeddings = self.molecule_encoder(batch["mol"])

        # batch['gsm'].x = self.molecule_encoder(batch['gsm'].x)

        # Encode the GSMs
        gsm_embeddings = self.gsm_encoder(batch["gsm"])
        # Calculate attention over gsm embeddings

        weights = F.softmax(molecule_embeddings, dim=1)  # (batch_size, num_pathways)
        pathways = torch.mm(
            batch["pathway_mask"], gsm_embeddings
        )  # MM[(num_pathways, num_nodes), (num_nodes, hidden_2)] = (num_pathways, hidden_2)
        output = torch.mm(
            pathways.t(), weights.t()
        )  # MM[(hidden_2, num_pathways), (num_pathways, batch_size)] = (hidden_2, batch_size)

        # Concatenate the embeddings
        embeddings = torch.cat([molecule_embeddings, output], dim=1)
        # Predict
        return self.MLP(embeddings)

    @staticmethod
    def add_args(parser) -> None:
        pass
