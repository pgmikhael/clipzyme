from typing import List, Literal
from nox.utils.registry import register_object
from nox.datasets.abstract import AbstractDataset
from nox.utils.pyg import from_smiles
from nox.utils.smiles import get_rdkit_feature
from nox.utils.proteins import get_protein_graphs_from_path
from tqdm import tqdm
import torch
import numpy as np
from collections import defaultdict
from nox.utils.messages import METAFILE_NOTFOUND_ERR, LOAD_FAIL_MSG
import traceback, warnings, os


@register_object("brenda_kcat", "dataset")
class BrendaKCat(AbstractDataset):
    def create_dataset(
        self, split_group: Literal["train", "dev", "test"]
    ) -> List[dict]:

        # get sequence,smiles pair to y-value
        seq_smi_2_y = defaultdict(list)
        for kcat_dict in self.metadata_json:
            seq_smi_2_y[f"{kcat_dict['Sequence']}{kcat_dict['Smiles']}"].append(
                self.get_label(kcat_dict)
            )

        # create samples
        dataset = []
        for kcat_dict in tqdm(self.metadata_json):

            if self.skip_sample(kcat_dict, split_group, seq_smi_2_y):
                continue
            mol_datapoint = from_smiles(kcat_dict["Smiles"])
            mol_datapoint.rdkit_features = torch.tensor(
                get_rdkit_feature(
                    kcat_dict["Smiles"], method=self.args.rdkit_features_name
                )
            )

            sample = {
                "mol": mol_datapoint,
                "smiles": kcat_dict["Smiles"],
                "sequence": kcat_dict["Sequence"],
                "y": self.get_label(kcat_dict),
                "sample_id": kcat_dict["sample_id"],
            }
            if self.args.generate_3d_graphs:
                sample["path"] = os.path.join(
                    "/Mounts/rbg-storage1/datasets/Enzymes/DLKcat/DeeplearningApproach/Data/database/Kcat_combination_0918_wildtype_mutant_structures/",
                    f"seq_id_{sample['seq_id']}.pdb",
                )

            dataset.append(sample)

        return dataset

    def get_label(self, sample):
        return np.log2(float(sample["Value"]))

    def skip_sample(self, sample, split_group, seq_smi_2_y) -> bool:
        """
        Return True if sample should be skipped and not included in data

        Ref: https://github.com/SysBioChalmers/DLKcat/blob/master/DeeplearningApproach/Code/model/preprocess_all.py
        """
        if sample["split"] != split_group:
            return True

        if "." in sample["Smiles"]:
            return True

        if float(sample["Value"]) <= 0:
            return True

        # skip if sequence, smile pair has inconsistent values (across organisms, conditions)
        smi = sample["Smiles"]
        seq = sample["Sequence"]
        if any(i != seq_smi_2_y[f"{seq}{smi}"][0] for i in seq_smi_2_y[f"{seq}{smi}"]):
            return True

        return False

    def assign_splits(self, metadata_json, split_probs, seed=0) -> None:
        np.random.seed(seed)
        if self.args.split_type == "random":
            super().assign_splits(metadata_json, split_probs, seed)

        elif self.args.split_type in ["smiles", "sequence", "ec", "organism"]:
            if self.args.split_type == "smiles":
                key = "Smiles"
            elif self.args.split_type == "sequence":
                key = "Sequence"
            elif self.args.split_type == "ec":
                key = "ECNumber"
            elif self.args.split_type == "organism":
                key = "Organism"

            #  split based on key
            samples = set([sample[key] for sample in metadata_json])
            samples = sorted(list(samples))
            np.random.shuffle(samples)
            split_indices = np.cumsum(np.array(split_probs) * len(samples)).astype(int)
            split_indices = np.concatenate([[0], split_indices])
            for i in range(len(split_indices) - 1):
                for sample in metadata_json:
                    if sample[key] in samples[split_indices[i] : split_indices[i + 1]]:
                        sample["split"] = ["train", "dev", "test"][i]

        elif self.args.split_type == "smiles_sequence":
            original_probs = split_probs
            split_probs = [split_probs[0] + split_probs[1], split_probs[2]]
            empirical_distribution = [-1, -1]
            while not np.allclose(
                empirical_distribution, split_probs, atol=0.1, rtol=0
            ):
                #  split based on sequence
                samples = set([sample["Sequence"] for sample in metadata_json])
                samples = sorted(list(samples))
                np.random.shuffle(samples)
                split_indices = np.cumsum(np.array(split_probs) * len(samples)).astype(
                    int
                )
                seq_split_indices = np.concatenate([[0], split_indices])

                #  split based on smiles
                samples = set([sample["Smiles"] for sample in metadata_json])
                samples = sorted(list(samples))
                np.random.shuffle(samples)
                split_indices = np.cumsum(np.array(split_probs) * len(samples)).astype(
                    int
                )
                smi_split_indices = np.concatenate([[0], split_indices])

                for i in range(len(split_indices) - 1):
                    for sample in metadata_json:
                        if (
                            sample["Smiles"]
                            in smi_split_indices[
                                split_indices[i] : split_indices[i + 1]
                            ]
                        ) and (
                            sample["Sequence"]
                            in seq_split_indices[
                                split_indices[i] : split_indices[i + 1]
                            ]
                        ):

                            sample["split"] = ["train", "test"][i]

                # compute empirical distribution
                empirical_distribution = [0, 0]
                for sample in metadata_json:
                    if sample["split"] == "train":
                        empirical_distribution[0] += 1
                    elif sample["split"] == "test":
                        empirical_distribution[1] += 1

            # assign dev set *randomly*
            dev_probs = original_probs[1] / split_probs[0]
            for sample in metadata_json:
                if sample["split"] == "train":
                    sample["split"] = np.random.choice(
                        ["train", "dev"], p=[1 - dev_probs, dev_probs]
                    )

        elif self.args.split_type == "mutation":
            #  split based on mutation
            assert len(split_probs) == 2, "Mutation split only supports 2 splits"
            for sample in metadata_json:
                if sample["Type"] == "wildtype":
                    sample["split"] = "train"
                else:
                    sample["split"] = np.random.choice(["dev", "test"], split_probs)
        else:
            raise ValueError("Split type not supported")

    def __getitem__(self, index):
        """
        Fetch single sample from dataset

        Args:
            index (int): random index of sample from dataset

        Returns:
            sample (dict): a sample
        """
        sample = self.dataset[index]
        if self.args.generate_3d_graphs:
            sample, data_params = get_protein_graphs_from_path([sample], self.args)
        try:
            return sample
        except Exception:
            warnings.warn(
                LOAD_FAIL_MSG.format(sample["sample_id"], traceback.print_exc())
            )

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        super(BrendaKCat, BrendaKCat).add_args(parser)

        parser.add_argument(
            "--rdkit_features_name",
            type=str,
            default="rdkit_fingerprint",
            help="name of rdkit features to use",
        )
        parser.add_argument(
            "--generate_3d_graphs",
            action="store_true",
            default=False,
            help="Generate 3D graphs from protein sequences",
        )
        parser.add_argument(
            "--protein_cache_path",
            type=str,
            default="/Mounts/rbg-storage1/datasets/Metabo/Brenda/cache",
            help="Path to cache protein graphs",
        )
        parser.add_argument(
            "--protein_resolution",
            type=str,
            default="residue",
            help="Resolution of protein graphs",
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            default=False,
            help="For debugging purposes, appends debug to cache paths",
        )
        parser.add_argument(
            "--no_graph_cache",
            action="store_true",
            default=False,
            help="Skip caching graphs",
        )
        parser.add_argument(
            "--knn_size",
            type=int,
            default=20,
            help="Number of nearest neighbors to use for graph construction",
        )

    @staticmethod
    def set_args(args) -> None:
        args.dataset_file_path = "/Mounts/rbg-storage1/datasets/Enzymes/DLKcat/DeeplearningApproach/Data/database/Kcat_combination_0918_wildtype_mutant_sample_ids.json"
        args.num_classes = 1

    @property
    def SUMMARY_STATEMENT(self) -> None:
        """
        Prints summary statement with dataset stats
        """
        num_substrates = len(set([s["smiles"] for s in self.dataset]))
        num_proteins = len(set([s["sequence"] for s in self.dataset]))
        statement = f""" 
        * Number of substrates: {num_substrates}
        * Number of proteins: {num_proteins}
        """
        return statement


@register_object("gsm_enzyme_interaction", "dataset")
class GSMInteraction(AbstractDataset):
    # load gsm json file
    # get (enzyme, reactant) pairs for each reaction with enzymes
    def create_dataset(
        self, split_group: Literal["train", "dev", "test"]
    ) -> List[dict]:
        dataset = []

        metabolites = set()
        enzyme2metabolites = defaultdict(set)
        id2sequence = {}

        for reaction in tqdm(self.metadata_json):
            if self.skip_sample(reaction=reaction):
                continue

            reactants = reaction["reactants"]
            enzymes = reaction.get("proteins", [])
            enzymes = [e for e in enzymes if not self.skip_sample(enzyme=e)]

            for enzyme in enzymes:
                for reactant in reactants:
                    sample = {
                        "smiles": reactant["smiles"],
                        "sequence": enzyme["protein_sequence"],
                        "enzyme_id": enzyme["bigg_gene_id"],
                        "metabolite_id": reactant["metabolite_id"],
                        "y": 1,
                    }
                    dataset.append(sample)
                    enzyme2metabolites[sample["enzyme_id"]].add(sample["metabolite_id"])
                    metabolites.add(sample["metabolite_id"])
                    id2sequence[sample["enzyme_id"]] = sample["sequence"]
                    id2sequence[sample["metabolite_id"]] = sample["smiles"]

        # sample negatives
        for enzyme_id, interacting_metabolites in enzyme2metabolites.items():
            noninteracting_metabolites = list(metabolites - interacting_metabolites)

            for metabolite in noninteracting_metabolites:
                sample = {
                    "smiles": id2sequence[metabolite],
                    "sequence": id2sequence[enzyme_id],
                    "enzyme_id": enzyme,
                    "metabolite_id": metabolite,
                    "y": 0,
                }
                dataset.append(sample)

        return dataset

    def skip_sample(self, **kwargs) -> bool:
        """
        Return True if sample should be skipped and not included in data
        """
        if kwargs.get("enzyme", False):
            # if missing protein sequence, skip sample
            if kwargs["enzyme"]["protein_sequence"] is None:
                return True

        if kwargs.get("reaction", False):
            # if is a pseudo-reaction (ie no reactants or products), skip sample
            if (
                len(kwargs["reaction"].get("reactants", [])) == 0
                or len(kwargs["reaction"].get("products", [])) == 0
            ):
                return True

            # biomass reaction is not true reaction
            if "BIOMASS" in kwargs["reaction"]["rxn_id"]:
                return True

        return False

    def assign_splits(self, metadata_json, split_probs, seed=0) -> None:
        np.random.seed(seed)
        if self.args.split_type == "random":
            super().assign_splits(metadata_json, split_probs, seed)

        elif self.args.split_type in ["sequence"]:
            # get all unique sequences
            sequences = set([s["sequence"] for s in metadata_json])

            # split sequences
            split_indices = np.random.choice(
                len(sequences), size=len(split_probs), replace=False, p=split_probs
            )
            split_indices.sort()

            # assign train/test split
            for i, sample in enumerate(metadata_json):
                for j in range(len(split_indices) - 1):
                    if (
                        sample["sequence"]
                        in sequences[split_indices[j] : split_indices[j + 1]]
                    ):
                        sample["split"] = ["train", "dev", "test"][j]

        else:
            raise ValueError("Split type not supported")
