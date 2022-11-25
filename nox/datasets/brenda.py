from typing import List, Literal
from nox.utils.registry import register_object
from nox.datasets.abstract import AbstractDataset
from nox.utils.pyg import from_smiles
from nox.utils.smiles import get_rdkit_feature
from nox.utils.amino_acids import AA_TO_SMILES
from tqdm import tqdm
import torch
import numpy as np
from collections import defaultdict
import argparse
import json
import os
from rxn.chemutils.smiles_randomization import randomize_smiles_rotated
import warnings
import hashlib


class Brenda(AbstractDataset):
    def load_dataset(self, args: argparse.ArgumentParser) -> None:
        super().load_dataset(args)
        self.metadata_json = self.metadata_json["data"]
        del self.metadata_json["spontaneous"]
        self.brenda_smiles = json.load(
            open(
                f"{os.path.dirname(args.dataset_file_path)}/brenda_substrates.json", "r"
            )
        )
        self.brenda_proteins = json.load(
            open(f"{os.path.dirname(args.dataset_file_path)}/brenda_proteins.json", "r")
        )

    def assign_splits(self, metadata_json, split_probs, seed=0) -> None:
        """
        Assigns each sample to a split group based on split_probs
        """
        # set seed
        np.random.seed(seed)
        if self.args.split_type in ["sequence", "ec"]:

            if self.args.split_type == "sequence":
                # split based on uniprot_id
                samples = []
                for _, ec_dict in metadata_json.items():
                    samples.extend(
                        v[0]["accessions"] for k, v in ec_dict["proteins"].items()
                    )

            elif self.args.split_type == "ec":
                # split based on ec number
                samples = list(metadata_json.keys())[1:]

            samples = sorted(list(set(samples)))
            np.random.shuffle(samples)
            split_indices = np.cumsum(np.array(split_probs) * len(samples)).astype(int)
            split_indices = np.concatenate([[0], split_indices])
            self.to_split = {}
            for i in range(len(split_indices) - 1):
                self.to_split.update(
                    {
                        sample: ["train", "dev", "test"][i]
                        for sample in samples[split_indices[i] : split_indices[i + 1]]
                    }
                )
        else:
            raise ValueError("Split type not supported")

    def get_proteinid_to_uniprot(self, ec_dict):

        proteinid2all_uniprot = {
            k: v[0]["accessions"] for k, v in ec_dict["proteins"].items()
        }
        proteinid2uniprot = {}
        for pid, uniprots in proteinid2all_uniprot.items():
            if len(uniprots) == 1:
                proteinid2uniprot[pid] = uniprots[0]
            else:
                if self.args.allow_multi_uniprots:
                    # if want to use with accessions that have multi uniprot ids, check if one of them self.brenda_proteins
                    uniprot_sequences = [
                        (uniprot, self.brenda_proteins.get(uniprot, None))
                        for uniprot in uniprots
                    ]
                    uniprot_sequences = [
                        x for x in uniprot_sequences if x[-1] is not None
                    ]
                    if len(uniprot_sequences) == 1:
                        proteinid2uniprot[pid] = uniprot_sequences[0][0]
                    elif len(set([x[-1] for x in uniprot_sequences])) == 1:
                        proteinid2uniprot[pid] = uniprot_sequences[0][0]
                    else:
                        continue

        return proteinid2uniprot

    def load_mcsa_data(self, args: argparse.ArgumentParser) -> dict:
        """Loads MCSA data

        Args:
            args (argparse.ArgumentParser): arguments

        Returns:
            dict: MCSA data {uniprot_id: {ec: [ residues ]}}
        """
        # protein: { reaction: [{residue: "", residueids: 0}] }
        mcsa_data = json.load(open(args.mcsa_file_path, "r"))

        protein2enzymatic_residues = {}
        for entry in mcsa_data:
            # all_ecs has length of 1
            ec = entry["all_ecs"][0]

            reference_uniprots = entry["reference_uniprot_id"].split(",")

            # there is only one uniprot and it is not empty
            unique_ref_uniprot_and_not_empty = (len(reference_uniprots) == 1) and (
                reference_uniprots[0] != ""
            )

            residue_uniprots = [
                residue["residue_sequences"][0]["uniprot_id"]
                for residue in entry["residues"]
            ]

            # at least one of the resiude uniprots missing uniprot
            empty_residue_uniprots = "" in residue_uniprots

            if empty_residue_uniprots and not unique_ref_uniprot_and_not_empty:
                continue  # skip entries with empty uniprot ids

            for residue in entry["residues"]:
                # residue['residue_chains']
                # residue["residue_sequences"] has length of 1
                uniprot = residue["residue_sequences"][0]["uniprot_id"]

                if empty_residue_uniprots:
                    uniprot = reference_uniprots[0]

                resid = residue["residue_sequences"][0]["resid"]
                amino_acid = residue["residue_sequences"][0]["code"]
                if uniprot not in protein2enzymatic_residues:
                    protein2enzymatic_residues[uniprot] = {ec: []}

                if ec not in protein2enzymatic_residues[uniprot]:
                    protein2enzymatic_residues[uniprot][ec] = []

                protein2enzymatic_residues[uniprot][ec].append(
                    {
                        "residue": amino_acid,
                        "residue_id": resid - 1,
                        "ec": ec,
                    }
                )

        return mcsa_data

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        super(Brenda, Brenda).add_args(parser)

        parser.add_argument(
            "--rdkit_features_name",
            type=str,
            default="rdkit_fingerprint",
            help="name of rdkit features to use",
        )
        parser.add_argument(
            "--enzyme_property",
            type=str,
            default=None,
            help="name of enzyme properties to use",
        )
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
        parser.add_argument(
            "--use_residues_in_reaction",
            action="store_true",
            default=False,
            help="Use residues as part of reaction string",
        )
        parser.add_argument(
            "--ec_level",
            type=int,
            default=0,
            choices=[0,1,2,3]
            help="EC level to use (e.g., ec_level 1 of '1.2.3.1' -> '1.2')",
        )

    @staticmethod
    def set_args(args) -> None:
        args.dataset_file_path = (
            "/Mounts/rbg-storage1/datasets/Enzymes/Brenda/brenda_2022_2.json"
        )

        args.mcsa_file_path = "/Mounts/rbg-storage1/datasets/Enzymes/MCSA/entries.json"


@register_object("brenda_constants", "dataset")
class BrendaConstants(Brenda):
    def create_dataset(
        self, split_group: Literal["train", "dev", "test"]
    ) -> List[dict]:

        samples = []
        for ec, ec_dict in tqdm(self.metadata_json.items(), desc="Creating dataset"):
            proteinid2uniprot = {
                k: v[0]["accessions"] for k, v in ec_dict["proteins"].items()
            }
            protein2organism = {k: v["value"] for k, v in ec_dict["organisms"].items()}

            for entry in ec_dict[self.args.enzyme_property]:
                proteins = entry.get("proteins", [])

                substrate = entry.get("value", None)

                for protein in proteins:

                    protein_id = proteinid2uniprot[protein]
                    organism = protein2organism[protein]

                    if "min_value" in entry and "max_value" in entry:
                        value = (
                            entry["min_value"],
                            entry["max_value"],
                        )
                    else:
                        value = entry["value"]

                    sample = {
                        "sequence": self.brenda_proteins[protein_id],
                        "protein_id": protein_id,
                        "y": self.get_label(value, self.args.enzyme_property),
                        "sample_id": f"org{organism}_ec{ec}_prot{protein_id}",
                        "ec": ec,
                        "organism": organism,
                    }

                    if substrate:
                        sample[
                            "sample_id"
                        ] = f"{sample['sample_id']}_substrate{substrate}"
                        sample["substrate"] = substrate
                        smiles = self.brenda_smiles.get(substrate, None)
                        if smiles:
                            mol_datapoint = from_smiles(smiles)
                            mol_datapoint.rdkit_features = torch.tensor(
                                get_rdkit_feature(
                                    smiles, method=self.args.rdkit_features_name
                                )
                            )
                            sample["mol"] = mol_datapoint
                        sample["smiles"] = smiles

                    samples.append(sample)

        # map (sequence, smile) pairs to list of labels
        seq_smi_2_y = defaultdict(list)
        for sample in samples:
            seq_smi_2_y[f"{sample['sequence']}{sample['smiles']}"].append(sample["y"])

        # filter through dataset
        dataset = []
        for sample in samples:
            if self.skip_sample(sample, seq_smi_2_y, split_group):
                continue
            dataset.append(sample)

        return dataset

    def skip_sample(self, sample, sequence_smiles2y, split_group) -> bool:
        # check right split
        if self.args.split_type == "sequence":
            if self.to_split[sample["protein_id"]] != split_group:
                return True

        if self.args.split_type == "ec":
            if self.to_split[sample["ec"]] != split_group:
                return True

        # check if sample has mol
        if sample["smiles"] is None:
            return True

        # if sequence is unknown
        if sample["sequence"] is None:
            return True

        # check if multiple sequences
        if len(sample["sequence"]) > 1:
            return True

        # check contradictory values TODO
        smi = sample["smiles"]
        seq = sample["sequence"]
        if any(
            i != sequence_smiles2y[f"{seq}{smi}"][0]
            for i in sequence_smiles2y[f"{seq}{smi}"]
        ):
            return True

        return False

    def get_label(self, value, property_name):
        if property_name == "turnover_number":
            return np.log2(value)
        elif property_name == "km_value":
            return np.log2(value)
        elif property_name == "ph_optimum":
            return value
        elif property_name == "specific_activity":
            return np.log2(value)
        elif property_name == "temperature_optimum":
            return value
        elif property_name == "isoelectric_point":
            return value
        elif property_name == "ki_value":
            return np.log2(value)
        elif property_name == "ic50":
            return np.log2(value)
        elif property_name == "kcat_km":
            return np.log2(value)
        elif property_name == "ph_stability":
            return value
        elif property_name == "temperature_stability":
            return (value - 55) / 24  # z-score
        elif property_name == "ph_range":
            return np.array(value)
        elif property_name == "temperature_range":
            return np.array(value)
        elif property_name == "localization":
            raise NotImplementedError
        elif property_name == "tissue":
            raise NotImplementedError
        raise ValueError(f"Property {property_name} not supported")


@register_object("brenda_ec", "dataset")
class BrendaEC(Brenda):
    def load_dataset(self, args: argparse.ArgumentParser) -> None:
        super().load_dataset(args)
        ecs = list(self.metadata_json.keys())
        ecs = sorted(list(set( [ '.'.join(e.split('.')[: self.args.ec_level + 1]) for e in ecs])))
        self.ec2class = {ec: i for i, ec in enumerate(ecs)}
        args.num_classes = len(ecs)


    def create_dataset(self, split_group: Literal["train", "dev", "test"]) -> List[dict]:
        # map uniprot to EC number (or sub-EC number)
        uniprot2ec = defaultdict(list)
        for ec, ec_dict in tqdm(
            self.metadata_json.items(), desc="Iterating over Brenda"
        ):
            ec_task = '.'.join(ec.split('.')[: self.args.ec_level + 1])

            for k, v in ec_dict["proteins"].items():
                uniprot2ec[v[0]["accessions"]].append(self.ec2class[ec_task])
        

        # create dataset of (protein, multi-task label) pairs
        dataset = []
        for protein_id, ec_list in tqdm(uniprot2ec.items(), desc="Creating dataset"):

            sample = {
                "sequence": self.brenda_proteins[protein_id],
                "protein_id": protein_id,
                "y": self.get_label(ec_list),
                "sample_id": f"prot{protein_id}",
            }

            if self.skip_sample(sample, split_group):
                continue

            dataset.append(sample)

        return dataset

    def get_label(self, ec_list):
        y = torch.zeros(len(self.args.num_classes))
        for ec in ec_list:
            y[ec] = 1
        return y

    def skip_sample(self, sample, split_group) -> bool:
        # check right split
        if self.to_split[sample["protein_id"]] != split_group:
            return True

        # if sequence is unknown
        if sample["sequence"] is None:
            return True

        # check if multiple sequences
        if len(sample["sequence"]) > 1:
            return True

        return False


@register_object("brenda_reaction", "dataset")
class BrendaReaction(Brenda):
    def create_dataset(
        self, split_group: Literal["train", "dev", "test"]
    ) -> List[dict]:

        mcsa_data = self.load_mcsa_data(self.args)

        # sort reactions

        uniprot2reactions = defaultdict(list)

        for ec, ec_dict in tqdm(self.metadata_json.items(), desc="Creating dataset"):
            proteinid2uniprot = {
                k: v[0]["accessions"] for k, v in ec_dict["proteins"].items()
            }
            protein2organism = {k: v["value"] for k, v in ec_dict["organisms"].items()}

            proteinid2uniprot = {
                k: v[0]["accessions"][0]
                for k, v in ec_dict["proteins"].items()
                if len(v[0]["accessions"]) == 1
            }
            for reaction_key in ["reaction", "natural_reaction"]:
                # reaction or natural_reaction may not exist
                if reaction_key in ec_dict:
                    for entry in ec_dict[reaction_key]:
                        # check both produces and reactants defined
                        if ("educts" in entry) and ("products" in entry):
                            # sort to check if reaction exists already
                            rs = sorted(entry["educts"])
                            ps = sorted(entry["products"])
                            reaction_string = ".".join(rs) + ">>" + ".".join(ps)
                            for protein in entry.get("proteins", []):
                                if proteinid2uniprot.get(protein, False):
                                    for uniprotid in proteinid2uniprot[protein]:
                                        catalogued_reactions = [
                                            rxn["reaction_string"]
                                            for rxn in uniprot2reactions[uniprotid]
                                        ]
                                        if reaction_string not in catalogued_reactions:

                                            sample_id = hashlib.md5(
                                                f"{uniprotid}_{reaction_string}".encode()
                                            ).hexdigest()

                                            residues = self.get_uniprot_residues(
                                                mcsa_data, uniprotid, ec
                                            )

                                            uniprot2reactions[uniprotid].append(
                                                {
                                                    "protein_id": uniprotid,
                                                    "sequence": self.brenda_proteins[
                                                        uniprotid
                                                    ],
                                                    "reactants": rs,
                                                    "products": ps,
                                                    "ec": ec,
                                                    "organism": protein2organism[
                                                        protein
                                                    ],
                                                    "reaction_string": ".".join(rs)
                                                    + ">>"
                                                    + ".".join(ps),
                                                    "sample_id": sample_id,
                                                    "residues": residues["residues"],
                                                    "residue_positions": residues[
                                                        "residue_mask"
                                                    ],
                                                    "has_residues": residues[
                                                        "has_residues"
                                                    ],
                                                }
                                            )

        # make each reaction a sample
        dataset = []
        for uniprot, reaction_list in uniprot2reactions.items():
            for reaction in reaction_list:
                if self.skip_sample(reaction, split_group):
                    continue
            dataset.append(reaction)

        return dataset

    def skip_sample(self, sample, split_group) -> bool:
        # check right split
        if self.args.split_type == "sequence":
            if self.to_split[sample["protein_id"]] != split_group:
                return True

        if self.args.split_type == "ec":
            if self.to_split[sample["ec"]] != split_group:
                return True

        # check if sample has mol
        if "?" in (sample["products"] + sample["reactants"]):
            return True

        # if sequence is unknown
        if sample["sequence"] is None:
            return True

        return False

    def get_uniprot_residues(self, mcsa_data, uniprotid, ec):
        """Get residues from MCSA data

        Args:
            mcsa_data (dict): MCSA data
            uniprotid (str): uniprot id
            ec (str): ec number

        Returns:
            torhc.Tensor: residue mask
        """
        sequence = self.brenda_proteins[uniprotid]
        y = torch.zeros(len(self.brenda_proteins[uniprotid]))
        has_y = 0
        residues = []
        if mcsa_data.get(uniprotid, False):
            if mcsa_data[uniprotid].get(ec, False):
                for residue_dict in mcsa_data[uniprotid][ec]:
                    y[residue_dict["residue_id"]] = 1
                    letter = sequence[residue_dict["residue_id"]]
                    residues.append(AA_TO_SMILES[letter])
                has_y = 1

        return {"residue_mask": y, "has_residues": has_y, "residues": residues}

    def __getitem__(self, index):
        try:
            sample = self.dataset[index]
            # augment: permute and/or randomize
            if (
                self.args.use_random_smiles_representation
                or self.args.randomize_order_in_reaction
            ):

                reactants, products = sample["reactants"], sample["products"]

            # incorporate sequence residues if known
            if self.args.use_residues_in_reaction:
                residues = sample["residues"]
                reactants.extend(residues)
                products.extend(residues)

            reaction = "{}>>{}".format(".".join(reactants), ".".join(products))
            # randomize order of reactants and products
            if self.args.randomize_order_in_reaction:
                np.random.shuffle(reactants)
                np.random.shuffle(products)
                reaction = "{}>>{}".format(".".join(reactants), ".".join(products))

            if self.args.use_random_smiles_representation:
                try:
                    reactants = [randomize_smiles_rotated(s) for s in reactants]
                    products = [randomize_smiles_rotated(s) for s in products]
                    reaction = "{}>>{}".format(".".join(reactants), ".".join(products))
                except:
                    pass

            item = {
                "reaction": reaction,
                "reactants": ".".join(reactants),
                "products": ".".join(products),
                "sequence": sample["sequence"],
                "ec": sample["ec"],
                "organism": sample["organism"],
                "protein_id": sample["protein_id"],
                "sample_id": sample["sample_id"],
                "residues": sample["residues"],
                "residue_positions": sample["residue_mask"],
                "has_residues": sample["has_residues"],
            }

            return item

        except Exception:
            warnings.warn(f"Could not load sample: {item['sample_id']}")
