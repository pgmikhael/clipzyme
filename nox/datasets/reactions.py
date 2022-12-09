import argparse
from typing import List, Literal
from nox.utils.registry import register_object, get_object
from nox.datasets.abstract import AbstractDataset
import warnings
from tqdm import tqdm
import random
from rxn.chemutils.smiles_randomization import randomize_smiles_rotated
from nox.utils.smiles import standardize_reaction
import copy 

@register_object("chemical_reactions", "dataset")
class ChemRXN(AbstractDataset):
    def create_dataset(
        self, split_group: Literal["train", "dev", "test"]
    ) -> List[dict]:
        dataset = []
        for rxn_dict in tqdm(self.metadata_json):
            if self.skip_sample(rxn_dict, split_group):
                continue
            dataset.append({"x": rxn_dict["reaction"], "sample_id": rxn_dict["rxnid"]})
        return dataset

    def skip_sample(self, sample, split_group) -> bool:
        if sample["split"] != split_group:
            return True
        return False

    def __getitem__(self, index):
        try:
            sample = copy.deepcopy(self.dataset[index])
            item = {}

            reaction = sample["x"]
            reactants, products = reaction.split(">>")
            reactants, products = reactants.split("."), products.split(".")

            # augment: permute and/or randomize
            if self.args.randomize_order_in_reaction and not (self.split_group == 'test'):
                random.shuffle(reactants)
                random.shuffle(products)
                reaction = "{}>>{}".format(".".join(reactants), ".".join(products))

            if self.args.use_random_smiles_representation and not (self.split_group == 'test'):
                try:
                    reactants = [randomize_smiles_rotated(s) for s in reactants]
                    products = [randomize_smiles_rotated(s) for s in products]
                    reaction = "{}>>{}".format(".".join(reactants), ".".join(products))
                except:
                    pass

            item["x"] = reaction
            item["reactants"] = ".".join(reactants)
            item["products"] = ".".join(products)
            
            if standardize_reaction(reaction) == ">>":
                return

            return item

        except Exception:
            warnings.warn(f"Could not load sample: {item['sample_id']}")

    @staticmethod
    def add_args(parser) -> None:
        super(ChemRXN, ChemRXN).add_args(parser)
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
