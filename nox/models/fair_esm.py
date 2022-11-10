import torch
import torch.nn as nn
import copy
from nox.models.abstract import AbstractModel
from nox.utils.classes import set_nox_type
from nox.utils.registry import register_object, get_object


@register_object("fair_esm", "model")
class FairEsm(AbstractModel):
    """
    Refer to https://github.com/facebookresearch/esm#available-models
    """
    def __init__(self, args):
        super(FairEsm, self).__init__()
        self.args = args
        torch.hub.set_dir(args.pretrained_hub_dir)
        self.model, self.alphabet = torch.hub.load(
            "facebookresearch/esm:main", args.esm_name
        )
        self.batch_converter = self.alphabet.get_batch_converter()
        self.register_buffer("devicevar", torch.zeros(1, dtype=torch.int8))
        if args.freeze_esm:
            self.model.eval()
        
        self.repr_layer = args.esm_hidden_layer

    def forward(self, x):
        """
        x: list of str (protein sequences)
        """
        output = {}
        fair_x = self.truncate_protein(x)
        batch_labels, batch_strs, batch_tokens = self.batch_converter(fair_x)
        batch_tokens = batch_tokens.to(self.devicevar.device)

        if self.args.freeze_esm:
            with torch.no_grad():
                result = self.model(
                    batch_tokens, repr_layers=[self.repr_layer], return_contacts=False
                )
        else:
            result = self.model(
                batch_tokens, repr_layers=[self.repr_layer], return_contacts=False
            )

        # Generate per-sequence representations via averaging
        hiddens = []
        for sample_num, sample in enumerate(x):
            hiddens.append(
                result["representations"][self.repr_layer][
                    sample_num, 1 : len(sample) + 1
                ].mean(0)
            )

        output["hidden"] = torch.stack(hiddens)

        return output
    
    def truncate_protein(self, x, max_length=1024):
        # max length allowed is 1024
        return [
            (i, s[: 1024 - 2]) if not isinstance(x[0], list) else (i, s[0][: 1024 - 2]) for i, s in enumerate(x) 
        ]  

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--pretrained_hub_dir",
            type=str,
            default="/Mounts/rbg-storage1/snapshots/metabolomics",
            help="directory to torch hub where pretrained models are saved",
        )
        parser.add_argument(
            "--esm_name",
            type=str,
            default="esm2_t12_35M_UR50D",
            help="directory to torch hub where pretrained models are saved",
        )
        parser.add_argument(
            "--freeze_esm",
            action="store_true",
            default=False,
            help="do not update encoder weights",
        )
        parser.add_argument(
            "--esm_hidden_layer",
            type=int,
            default=12,
            help="do not update encoder weights",
        )

@register_object("fair_esm2", "model")
class FairEsm2(FairEsm):
    def truncate_protein(self, x, max_length=torch.inf):
        return [
            (i, s) if not isinstance(x[0], list) else (i, s[0]) for i, s in enumerate(x) 
        ]  
    

@register_object("protein_encoder", "model")
class ProteinEncoder(AbstractModel):
    def __init__(self, args):
        super(ProteinEncoder, self).__init__()
        self.args = args
        self.encoder = get_object(args.protein_encoder_type, "model")(args)
        cargs = copy.deepcopy(args)
        cargs.mlp_input_dim = args.protein_hidden_dim
        args.freeze = args.freeze_encoder
        self.mlp = get_object(args.protein_classifer, "model")(cargs)
        if self.args.freeze_encoder:
            self.encoder.eval()

    def forward(self, batch):
        output = {}
        if self.args.freeze_encoder:
            with torch.no_grad():
                output_esm = self.encoder(batch["x"])
        else:
            output_esm = self.encoder(batch["x"])
        # output["protein_hidden"] = output_esm["hidden"]
        output.update( 
                self.mlp({"x": output_esm["hidden"]}) 
                )
        return output

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--protein_encoder_type",
            type=str,
            default="fair_esm",
            help="name of the protein encoder",
            action=set_nox_type("model"),
        )
        parser.add_argument(
            "--freeze_encoder",
            action="store_true",
            default=False,
            help="do not update encoder weights",
        )
        parser.add_argument(
            "--protein_hidden_dim",
            type=int,
            default=480,
            help="hidden dimension of the protein",
        )
        parser.add_argument(
            "--protein_classifer",
            type=str,
            default="mlp_classifier",
            help="name of classifier",
            action=set_nox_type("model"),
        )
