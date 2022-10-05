import argparse
from typing import List, Literal
from nox.utils.registry import register_object
from nox.utils.smiles import tokenize_smiles
from nox.datasets.abstract import AbstractDataset
import warnings
from tqdm import tqdm
import random
from rxn.chemutils.smiles_randomization import randomize_smiles_rotated


@register_object("chemical_reactions", "dataset")
class ChemRXN(AbstractDataset):
    def create_dataset(
        self, split_group: Literal["train", "dev", "test"]
    ) -> List[dict]:
        dataset = []
        for rxn_dict in tqdm(self.metadata_json):
            if self.skip_sample(rxn_dict, split_group):
                continue
            dataset.append({"x": rxn_dict["reaction"]})
        return dataset

    def skip_sample(self, sample, split_group) -> bool:
        if sample["split"] != split_group:
            return True
        return False

    def __getitem__(self, index):
        try:
            item = self.dataset[index]
            # augment: permute and/or randomize
            if (
                self.args.use_random_smiles_representation
                or self.args.randomize_order_in_reaction
            ):
                reaction = item["x"]
                reactants, products = reaction.split(">>")
                reactants, products = reactants.split("."), products.split(".")

            if self.args.randomize_order_in_reaction:
                random.shuffle(reactants)
                random.shuffle(products)
                reaction = "{}>>{}".format(".".join(reactants), ".".join(products))

            if self.args.use_random_smiles_representation:
                reactants = [randomize_smiles_rotated(s) for s in reactants]
                products = [randomize_smiles_rotated(s) for s in products]
                reaction = "{}>>{}".format(".".join(reactants), ".".join(products))

            item["x"] = reaction

            return item

        except Exception:
            warnings.warn("Could not load sample")

    @staticmethod
    def add_args(parser) -> None:
        super().add_args(parser)
        parser.add_argument(
            "--randomize_order_in_reaction",
            action="store_true",
            default=False,
            help="Permute smiles in reactants and in products as augmentation",
        )
        parser.add_argument(
            "--use_random_smiles_representation",
            action="store_true",
            default=False,
            help="Use non-canonical representation of smiles as augmentation",
        )
