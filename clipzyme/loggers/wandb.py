from clipzyme.utils.registry import register_object
import pytorch_lightning as pl
import os
from clipzyme.utils.classes import Nox
import wandb
import rdkit


@register_object("wandb", "logger")
class WandB(pl.loggers.WandbLogger, Nox):
    def __init__(self, args) -> None:
        super().__init__(
            project=args.project_name,
            name=args.experiment_name,
            entity=args.workspace,
            tags=args.logger_tags,
        )

    def setup(self, **kwargs):
        # "gradients", "parameters", "all", or None
        # # change "log_freq" log frequency of gradients and parameters (100 steps by default)
        if kwargs["args"].local_rank == 0:
            self.watch(kwargs["model"], log="all")
            self.experiment.config.update(kwargs["args"])

    def log_image(self, image, name):
        self.log_image(images=[image], caption=[name])

    def log_table(self, table_columns, table_data, data_types, table_name):
        """
        table_columns: list of column names
        table_data: list per column
        data_types: list of data type per column
        """
        formatted_data = []
        for datalist, datatype in zip(table_data, data_types):
            if datatype == "image":
                formatted_data.append([wandb.Image(dataitem) for dataitem in datalist])
            elif datatype == "video":
                formatted_data.append(
                    [
                        wandb.Video(dataitem, format="gif", fps=4)
                        for dataitem in datalist
                    ]
                )
            elif datatype == "smiles":
                fd = []
                for dataitem in datalist:
                    mol = rdkit.Chem.Draw.MolToImage(
                        rdkit.Chem.MolFromSmiles(dataitem), size=(200, 200)
                    )
                    item = wandb.Image(mol)
                    fd.append(item)
                formatted_data.append(fd)
            else:
                formatted_data.append(datalist)

        long_format = list(zip(*formatted_data))  # transform into list of rows

        predictions_table = wandb.Table(columns=table_columns, data=long_format)
        self.experiment.log({table_name: predictions_table})
