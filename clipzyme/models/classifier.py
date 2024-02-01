import torch
import torch.nn as nn
import copy
from clipzyme.utils.registry import register_object, get_object
from clipzyme.utils.classes import set_nox_type
from clipzyme.models.abstract import AbstractModel


@register_object("classifier", "model")
class Classifier(AbstractModel):
    def __init__(self, args):
        super(Classifier, self).__init__()

        self.args = args
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 2)))
        if args.pretrained_encoder_path:
            state_dict = torch.load(args.pretrained_encoder_path)
            self.encoder = get_object(args.model_name_for_encoder, "model")(
                state_dict["hyper_parameters"]["args"]
            )
            self.encoder.load_state_dict(
                {k[len("model.") :]: v for k, v in state_dict["state_dict"].items()}
            )
            self.encoder.args = args
        else:
            self.encoder = get_object(args.model_name_for_encoder, "model")(args)

        cargs = copy.deepcopy(args)
        cargs.mlp_input_dim = args.classifier_mlp_input_dim
        cargs.mlp_layer_configuration = args.classifier_mlp_layer_configuration
        cargs.mlp_use_batch_norm = args.classifier_mlp_use_batch_norm
        cargs.mlp_use_layer_norm = args.classifier_mlp_use_layer_norm
        self.mlp = get_object("mlp_classifier", "model")(cargs)
        self.encoder_hidden_key = args.encoder_hidden_key

    def forward(self, batch=None):
        output = {}
        output["encoder_hidden"] = self.encoder(batch)[self.encoder_hidden_key]
        output.update(self.mlp({"x": output["encoder_hidden"]}))
        return output

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--model_name_for_encoder",
            type=str,
            action=set_nox_type("model"),
            default="resnet18",
            help="Name of encoder to use",
        )
        parser.add_argument(
            "--pretrained_encoder_path",
            type=str,
            default=None,
            help="path to pretrained if applicable",
        )
        parser.add_argument(
            "--encoder_hidden_key",
            type=str,
            default="hidden",
            help="name of hidden features from encoder output",
        )
        parser.add_argument(
            "--classifier_mlp_input_dim",
            type=int,
            default=512,
            help="Dim of input to mlp",
        )
        parser.add_argument(
            "--classifier_mlp_layer_configuration",
            type=int,
            nargs="*",
            default=[],
            help="MLP layer dimensions",
        )
        parser.add_argument(
            "--classifier_mlp_use_batch_norm",
            action="store_true",
            default=False,
            help="Use batchnorm in mlp",
        )
        parser.add_argument(
            "--classifier_mlp_use_layer_norm",
            action="store_true",
            default=False,
            help="Use LayerNorm in mlp",
        )


@register_object("mlp_classifier", "model")
class MLPClassifier(AbstractModel):
    def __init__(self, args):
        super(MLPClassifier, self).__init__()

        self.args = args

        model_layers = []
        cur_dim = args.mlp_input_dim
        for layer_size in args.mlp_layer_configuration:
            model_layers.extend(self.append_layer(cur_dim, layer_size, args))
            cur_dim = layer_size

        self.mlp = nn.Sequential(*model_layers)
        self.predictor = nn.Linear(cur_dim, args.num_classes)

    def append_layer(self, cur_dim, layer_size, args, with_dropout=True):
        linear_layer = nn.Linear(cur_dim, layer_size)
        bn = nn.BatchNorm1d(layer_size)
        ln = nn.LayerNorm(layer_size)
        if args.mlp_use_batch_norm:
            seq = [linear_layer, bn, nn.ReLU()]
        elif args.mlp_use_layer_norm:
            seq = [linear_layer, ln, nn.ReLU()]
        else:
            seq = [linear_layer, nn.ReLU()]
        if with_dropout:
            seq.append(nn.Dropout(p=args.dropout))
        return seq

    def forward(self, batch=None):
        output = {}
        z = self.mlp(batch["x"])
        output["hidden"] = z
        output["logit"] = self.predictor(z)
        return output

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--mlp_input_dim", type=int, default=512, help="Dim of input to mlp"
        )
        parser.add_argument(
            "--mlp_layer_configuration",
            type=int,
            nargs="*",
            default=[],
            help="MLP layer dimensions",
        )
        parser.add_argument(
            "--mlp_use_batch_norm",
            action="store_true",
            default=False,
            help="Use batchnorm in mlp",
        )
        parser.add_argument(
            "--mlp_use_layer_norm",
            action="store_true",
            default=False,
            help="Use LayerNorm in mlp",
        )


@register_object("graph_classifier", "model")
class GraphClassifier(Classifier):
    def __init__(self, args):
        super(GraphClassifier, self).__init__(args)
        cargs = copy.deepcopy(args)
        cargs.mlp_layer_configuration = args.graph_classifier_mlp_layer_configuration
        cargs.mlp_use_batch_norm = args.classifier_mlp_use_batch_norm
        cargs.mlp_use_layer_norm = args.classifier_mlp_use_layer_norm
        if self.args.use_rdkit_features:
            cargs.mlp_input_dim = (
                args.graph_classifier_hidden_dim + args.rdkit_features_dim
            )
        self.mlp = get_object("mlp_classifier", "model")(cargs)

    def forward(self, batch=None):
        output = {}
        graph_x = self.encoder(batch)[self.encoder_hidden_key]
        output["encoder_hidden"] = graph_x
        batch_size = output["encoder_hidden"].shape[0]
        if self.args.use_rdkit_features:
            features = batch["rdkit_features"].view(batch_size, -1)
            graph_x = torch.concat([graph_x, features], dim=-1)
        output["hidden"] = graph_x
        output.update(self.mlp({"x": graph_x.float()}))
        return output

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        super(GraphClassifier, GraphClassifier).add_args(parser)
        parser.add_argument(
            "--use_rdkit_features",
            action="store_true",
            default=False,
            help="whether using graph-level features from rdkit",
        )
        parser.add_argument(
            "--rdkit_features_dim",
            type=int,
            default=0,
            help="number of features",
        )
        parser.add_argument(
            "--graph_classifier_hidden_dim",
            type=int,
            default=None,
            help="dimension of hidden layer from graph encoder",
        )
        parser.add_argument(
            "--graph_classifier_mlp_layer_configuration",
            type=int,
            nargs="*",
            default=[],
            help="MLP layer dimensions",
        )
