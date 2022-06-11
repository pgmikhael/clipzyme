from nox.utils.registry import register_object
from pytorch_lightning.callbacks import StochasticWeightAveraging
from nox.utils.classes import Nox


@register_object("swa", "callback")
class SWA(StochasticWeightAveraging, Nox):
    def __init__(self, args) -> None:
        if "." in args.swa_epoch:
            swa_epoch = float(args.swa_epoch)
        else:
            swa_epoch = int(args.swa_epoch)

        super().__init__(
            swa_epoch_start=swa_epoch,
            swa_lrs=args.swa_lr,
            annealing_epochs=args.swa_annealing_epochs,
            annealing_strategy=args.swa_annealing_strategy,
            avg_fn=None,
        )

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        # stochastic weight averaging
        parser.add_argument(
            "--swa_epoch",
            type=str,
            default="0.8",
            help="when to start swa",
        )

        parser.add_argument(
            "--swa_lr",
            type=float,
            default=None,
            help="lr for swa. None will use existing lr",
        )
        parser.add_argument(
            "--swa_annealing_epochs",
            type=int,
            default=10,
            help="number of epochs in the annealing phase",
        )
        parser.add_argument(
            "--swa_annealing_strategy",
            type=str,
            choices=["cos", "linear"],
            default="cos",
            help="lr annealing strategy",
        )
