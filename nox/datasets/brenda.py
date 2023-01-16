from typing import List, Literal
from nox.utils.registry import register_object
from nox.datasets.abstract import AbstractDataset
from nox.utils.pyg import from_smiles
from nox.utils.smiles import get_rdkit_feature
from nox.utils.amino_acids import AA_TO_SMILES
from nox.utils.proteins import get_protein_graphs_from_path
from tqdm import tqdm
import torch
import numpy as np
from collections import defaultdict
import argparse
import json
import os
from rxn.chemutils.smiles_randomization import randomize_smiles_rotated
import traceback, warnings
import hashlib
from frozendict import frozendict
import copy
from rich import print as rprint
from nox.utils.messages import METAFILE_NOTFOUND_ERR, LOAD_FAIL_MSG

CHEBI_DB = json.load(open("/Mounts/rbg-storage1/datasets/Metabo/chebi_db.json", "r"))


def add_mcsa_data(
    protein2enzymatic_residues,
    mcsa_curated_proteins,
    sequence,
    uniprot,
    ec,
    reactants,
    products,
    amino_acid,
    resid,
    is_reference,
):
    # add residue and sequence information
    if sequence not in protein2enzymatic_residues:
        protein2enzymatic_residues[sequence] = {
            ec: {
                "residues": [],
                "reactants": reactants,
                "products": products,
            },
            "sequence": sequence,
            "uniprot": uniprot,
        }

    if ec not in protein2enzymatic_residues[sequence]:
        protein2enzymatic_residues[sequence][ec] = {
            "residues": [],
            "reactants": reactants,
            "products": products,
        }

    sample = frozendict(
        {
            "residue": amino_acid,
            "residue_id": resid - 1,
            "ec": ec,
            "is_reference": is_reference,
        }
    )

    if sample not in protein2enzymatic_residues[sequence][ec]["residues"]:
        protein2enzymatic_residues[sequence][ec]["residues"].append(sample)

    if is_reference and not (sequence in mcsa_curated_proteins):
        mcsa_curated_proteins[sequence] = True  # store as dict for faster lookup
    return protein2enzymatic_residues, mcsa_curated_proteins


class Brenda(AbstractDataset):
    def load_dataset(self, args: argparse.ArgumentParser) -> None:
        super().load_dataset(args)
        self.metadata_json = self.metadata_json["data"]
        del self.metadata_json["spontaneous"]
        self.brenda_smiles = json.load(
            open(f"{os.path.dirname(args.dataset_file_path)}/brenda_smiles.json", "r")
        )
        self.brenda_proteins = json.load(
            open(f"{os.path.dirname(args.dataset_file_path)}/brenda_proteins.json", "r")
        )

    def assign_splits(self, metadata_json, split_probs, seed=0) -> None:
        """
        Assigns each sample to a split group based on split_probs
        """
        # get all samples
        rprint("Generating dataset in order to assign splits...")
        dataset = self.create_dataset(
            "train"
        )  # must not skip samples by using split in dataset
        self.to_split = {}

        # set seed
        np.random.seed(seed)

        # assign groups
        if self.args.split_type in ["sequence", "ec", "product"]:

            if self.args.split_type == "sequence":
                # split based on uniprot_id
                samples = [s["protein_id"] for s in dataset]

            elif self.args.split_type == "ec":
                # split based on ec number
                samples = [s["ec"] for s in dataset]

                # option to change level of ec categorization based on which to split
                samples = [
                    ".".join(e.split(".")[: self.args.ec_level + 1]) for e in samples
                ]

            elif self.args.split_type == "product":
                # split by reaction product (splits share no products)
                if any(len(s["products"]) > 1 for s in dataset):
                    raise NotImplementedError(
                        "Product split not implemented for multi-products"
                    )

                samples = [p for s in dataset for p in s["products"]]

            samples = sorted(list(set(samples)))
            np.random.shuffle(samples)
            split_indices = np.cumsum(np.array(split_probs) * len(samples)).astype(int)
            split_indices = np.concatenate([[0], split_indices])

            for i in range(len(split_indices) - 1):
                self.to_split.update(
                    {
                        sample: ["train", "dev", "test"][i]
                        for sample in samples[split_indices[i] : split_indices[i + 1]]
                    }
                )

        # random splitting
        elif self.args.split_type == "random":

            for sample in dataset:
                seq = sample["sequence"]
                smi = sample["smiles"]
                self.to_split.update(
                    {
                        f"{seq}{smi}": np.random.choice(
                            ["train", "dev", "test"], p=split_probs
                        )
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
            dict: MCSA data {uniprot_id: {ec: [ residues ], sequence = ""}}
        """

        mcsa_biomolecules = json.load(open(args.mcsa_biomolecules_path, "r"))
        mcsa_curated_data = json.load(open(args.mcsa_file_path, "r"))

        # load data
        pdb2uniprot = json.load(open(args.mcsa_pdb_to_uniprots, "r"))
        mcsa_homologs = json.load(open(args.mcsa_homologs_file_path, "r"))
        mcsa_molecules = mcsa_biomolecules["molecules"]
        mcsa_proteins = mcsa_biomolecules["proteins"]

        # to store reference proteins and not add them twice when processing homologs
        mcsa_curated_proteins = {}
        protein2enzymatic_residues = {}
        for entry in tqdm(mcsa_curated_data, desc="Processing M-CSA data"):
            # all_ecs has length of 1
            ec = entry["all_ecs"][0]

            # reaction
            reaction = entry["reaction"]["compounds"]
            reactants = [
                mcsa_molecules[c["chebi_id"]].get("SMILES", None)
                for c in reaction
                if (mcsa_molecules[c["chebi_id"]]) and (c["type"] == "reactant")
            ]
            products = [
                mcsa_molecules[c["chebi_id"]].get("SMILES", None)
                for c in reaction
                if (mcsa_molecules[c["chebi_id"]]) and (c["type"] == "product")
            ]

            # get reference (curated) uniprot data
            for residue in entry["residues"]:

                # loop over uniprot sequences
                for seq in residue["residue_sequences"]:
                    uniprot = seq["uniprot_id"]
                    resid = seq["resid"]  # residue position
                    amino_acid = seq["code"]  # residue 3-letter code

                    if any(k in ["", None] for k in [uniprot, resid]):
                        continue

                    if not mcsa_proteins.get(uniprot, False):
                        continue

                    if not mcsa_proteins[uniprot].get(uniprot, False):
                        continue

                    sequence = mcsa_proteins[uniprot][uniprot]["sequence"]  # sequence

                    protein2enzymatic_residues, mcsa_curated_proteins = add_mcsa_data(
                        protein2enzymatic_residues,
                        mcsa_curated_proteins,
                        sequence,
                        uniprot,
                        ec,
                        reactants,
                        products,
                        amino_acid,
                        resid,
                        is_reference=True,
                    )

                # loop over pdb chains
                for chain in residue["residue_chains"]:
                    pdb = chain["pdb_id"]
                    assembly = chain.get("assembly", None)

                    for uniprot_dict in pdb2uniprot.get(pdb, []):
                        uniprot = uniprot_dict["uniprot"]
                        assembly_name = f"{uniprot}-{assembly}"
                        if assembly is None:
                            assembly_name = uniprot

                        resid = chain["resid"]  # residue position
                        amino_acid = chain["code"]  # residue 3-letter code

                        if any(k in ["", None] for k in [uniprot, resid]):
                            continue

                        if not mcsa_proteins.get(uniprot, False):
                            continue

                        if not mcsa_proteins[uniprot].get(assembly_name, False):
                            continue

                        sequence = mcsa_proteins[uniprot][assembly_name][
                            "sequence"
                        ]  # sequence

                        (
                            protein2enzymatic_residues,
                            mcsa_curated_proteins,
                        ) = add_mcsa_data(
                            protein2enzymatic_residues,
                            mcsa_curated_proteins,
                            sequence,
                            assembly_name,
                            ec,
                            reactants,
                            products,
                            amino_acid,
                            resid,
                            is_reference=True,
                        )

            # process homologs in similar fashion
            homologs = [m for m in mcsa_homologs if m["mcsa_id"] == entry["mcsa_id"]]

            # add homologs to dataset
            for homolog_residues in homologs:

                for homolog_entry in homolog_residues["residue_sequences"]:
                    uniprot = homolog_entry["uniprot_id"]
                    resid = homolog_entry["resid"]
                    amino_acid = homolog_entry["code"]

                    if any(k in ["", None] for k in [uniprot, resid]):
                        continue

                    if not mcsa_proteins.get(uniprot, False):
                        continue

                    if not mcsa_proteins[uniprot].get(uniprot, False):
                        continue

                    sequence = mcsa_proteins[uniprot][uniprot]["sequence"]  # sequence

                    is_reference = sequence in mcsa_curated_proteins
                    if is_reference:
                        continue  # skip reference entry, already added above (would not need above if homologs contained all references)

                    protein2enzymatic_residues, mcsa_curated_proteins = add_mcsa_data(
                        protein2enzymatic_residues,
                        mcsa_curated_proteins,
                        sequence,
                        uniprot,
                        ec,
                        reactants,
                        products,
                        amino_acid,
                        resid,
                        is_reference=False,
                    )

                for assembly_entry in homolog_residues["residue_chains"]:
                    assembly = assembly_entry["assembly"]
                    pdb = chain["pdb_id"]

                    for uniprot_dict in pdb2uniprot.get(pdb, []):
                        uniprot = uniprot_dict["uniprot"]

                        assembly_name = f"{uniprot}-{assembly}"
                        if assembly is None:
                            assembly_name = uniprot

                        resid = chain["resid"]  # residue position
                        amino_acid = chain["code"]  # residue 3-letter code

                        if any(k in ["", None] for k in [uniprot, resid]):
                            continue

                        if not mcsa_proteins.get(uniprot, False):
                            continue

                        if not mcsa_proteins[uniprot].get(assembly_name, False):
                            continue

                        sequence = mcsa_proteins[uniprot][assembly_name][
                            "sequence"
                        ]  # sequence

                        is_reference = sequence in mcsa_curated_proteins
                        if is_reference:
                            continue  # skip reference entry, already added above (would not need above if homologs contained all references)

                        (
                            protein2enzymatic_residues,
                            mcsa_curated_proteins,
                        ) = add_mcsa_data(
                            protein2enzymatic_residues,
                            mcsa_curated_proteins,
                            sequence,
                            assembly_name,
                            ec,
                            reactants,
                            products,
                            amino_acid,
                            resid,
                            is_reference=False,
                        )

        return protein2enzymatic_residues

    def get_smiles(self, substrate):
        substrate_data = self.brenda_smiles.get(substrate, None)
        if substrate_data is None:
            return
        if substrate_data.get("chebi_data", False):
            return substrate_data["chebi_data"].get("SMILES", None)
        elif substrate_data.get("pubchem_data", False):
            if isinstance(substrate_data["pubchem_data"], dict):
                return substrate_data["pubchem_data"].get("CanonicalSMILES", None)
            elif isinstance(substrate_data["pubchem_data"], list):
                return substrate_data["pubchem_data"][0].get("CanonicalSMILES", None)
            else:
                raise NotImplementedError
        return

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
            default=3,
            choices=[0, 1, 2, 3],
            help="EC level to use (e.g., ec_level 1 of '1.2.3.1' -> '1.2')",
        )
        parser.add_argument(
            "--deduplicate_reactions",
            action="store_true",
            default=False,
            help="Create reaction dataset of unique chemical reactions. Used to skip same reaction for different proteins",
        )
        parser.add_argument(
            "--mcsa_file_path",
            type=str,
            default="/Mounts/rbg-storage1/datasets/Enzymes/MCSA/entries.json",
            help="M-CSA entries data",
        )
        parser.add_argument(
            "--mcsa_homologs_file_path",
            type=str,
            default="/Mounts/rbg-storage1/datasets/Enzymes/MCSA/homologues_residues.json",
            help="M-CSA homologues entries data",
        )
        parser.add_argument(
            "--mcsa_biomolecules_path",
            type=str,
            default="/Mounts/rbg-storage1/datasets/Enzymes/MCSA/mcsa_biomolecules.json",
            help="M-CSA biomolecules metadata",
        )
        parser.add_argument(
            "--mcsa_pdb_to_uniprots",
            type=str,
            default="/Mounts/rbg-storage1/datasets/Enzymes/MCSA/pdb2uniprotlite.json",
            help="M-CSA biomolecules metadata",
        )
        parser.add_argument(
            "--mcsa_skip_unk_smiles",
            action="store_true",
            default=False,
            help="Skip entries with unknown smiles",
        )
        parser.add_argument(
            "--precomputed_esm_features_dir",
            type=str,
            default=None,
            help="directory with precomputed esm features for computation efficiency",
        )
        parser.add_argument(
            "--max_protein_length",
            type=int,
            default=None,
            help="skip proteins longer than max_protein_length",
        )

    @staticmethod
    def set_args(args) -> None:
        super(Brenda, Brenda).set_args(args)
        args.dataset_file_path = (
            "/Mounts/rbg-storage1/datasets/Enzymes/Brenda/brenda_2022_2.json"
        )


@register_object("brenda_constants", "dataset")
class BrendaConstants(Brenda):
    def create_dataset(
        self, split_group: Literal["train", "dev", "test"]
    ) -> List[dict]:

        samples = []
        for ec, ec_dict in tqdm(self.metadata_json.items(), desc="Creating dataset"):
            if not ec_dict.get("proteins", False):
                continue

            proteinid2uniprot = {
                k: v[0]["accessions"] for k, v in ec_dict["proteins"].items()
            }
            protein2organism = {k: v["value"] for k, v in ec_dict["organisms"].items()}

            for entry in ec_dict.get(self.args.enzyme_property, []):
                proteins = entry.get("proteins", [])

                substrate = entry.get("value", None)

                for protein in proteins:

                    protein_ids = proteinid2uniprot[protein]
                    if isinstance(protein_ids, str):
                        protein_ids = [protein_ids]
                    organism = protein2organism[protein].replace(" ", "_")

                    if "min_value" in entry and "max_value" in entry:
                        if not self.args.use_mean_labels:
                            value = (
                                entry["min_value"],
                                entry["max_value"],
                            )
                        else:
                            value = np.mean(
                                [
                                    entry["min_value"],
                                    entry["max_value"],
                                ]
                            )
                    elif not "num_value" in entry:
                        if "min_value" in entry:
                            value = entry["min_value"]
                        elif "max_value" in entry:
                            value = entry["max_value"]
                        else:
                            print("Skipped because no value found for entry ", entry)
                            continue
                    else:
                        value = entry["num_value"]

                    for protein_id in protein_ids:
                        sample = {
                            "sequence": self.brenda_proteins[protein_id]["sequence"],
                            "protein_id": protein_id,
                            "y": self.get_label(value, self.args.enzyme_property),
                            "sample_id": f"org{organism.lower()}_ec{ec}_prot{protein_id}",
                            "ec": ec,
                            "organism": organism,
                        }

                        if substrate:
                            sample[
                                "sample_id"
                            ] = f"{sample['sample_id']}_substrate{substrate}"
                            sample["substrate"] = substrate
                            smiles = self.get_smiles(substrate)
                            if smiles:
                                try:
                                    mol_datapoint = from_smiles(smiles)
                                    mol_datapoint.rdkit_features = torch.tensor(
                                        get_rdkit_feature(
                                            smiles, method=self.args.rdkit_features_name
                                        )
                                    )
                                    sample["mol"] = mol_datapoint
                                except:
                                    print(
                                        "Skipped sample because could not convert smiles to RDKit Mol"
                                    )
                                    continue
                            sample["smiles"] = smiles

                        samples.append(sample)

        # map (sequence, smile) pairs to list of labels
        seq_smi_2_y = defaultdict(list)
        for sample in samples:
            seq_smi_2_y[f"{sample['sequence']}{sample['smiles']}"].append(sample["y"])

        # filter through dataset
        dataset = []
        samples_added = set()
        for sample in samples:
            if self.skip_sample(sample, seq_smi_2_y, split_group):
                continue

            if self.args.use_mean_labels:
                # it is was not necessary for kcat_km
                seq = sample["sequence"]
                smi = sample["smiles"]
                # keep track of samples added to avoid duplicates

                # either have multiple same samples with different labels
                # each identical sample could either have a numpy array label or a float label
                different_labels_bool = any(
                    not np.array_equal(i, seq_smi_2_y[f"{seq}{smi}"][0])
                    if isinstance(i, np.ndarray)
                    else i != seq_smi_2_y[f"{seq}{smi}"][0]
                    for i in seq_smi_2_y[f"{seq}{smi}"]
                )
                # or I have a multi-label label (ie range)
                multiple_labels_bool = (
                    not isinstance(sample["y"], float) and len(sample["y"]) > 1
                ) or (isinstance(sample["y"], np.ndarray) and len(sample["y"]) > 1)

                # if I have already added this sample, skip it
                if f"{seq}{smi}" in samples_added:
                    continue

                if different_labels_bool or multiple_labels_bool:
                    # in which case mean the labels
                    labels = []
                    for i in seq_smi_2_y[f"{seq}{smi}"]:
                        if isinstance(i, np.ndarray):
                            labels.append(np.mean(i))
                        elif (
                            not isinstance(i, float) and len(i) > 1
                        ):  # probably a tuple
                            labels.append(np.mean(i))
                        else:
                            labels.append(i)  # label is just a float
                    sample["y"] = float(np.mean(labels))

                samples_added.add(f"{seq}{smi}")

            dataset.append(sample)

        return dataset

    def skip_sample(self, sample, sequence_smiles2y, split_group) -> bool:
        # check if sample has mol
        if sample["smiles"] is None:
            # print("Skipped sample because SMILE is None")
            return True

        # if sequence is unknown
        if sample["sequence"] is None:
            print("Skipped sample because Sequence is None")
            return True

        # check if multiple sequences
        # if len(sample["sequence"]) > 1: # each sample is a single sequence
        # return True

        # check either all labels are multi value or single value
        if self.args.enzyme_property == "turnover_number":
            if not self.args.use_mean_labels and (
                isinstance(sample["y"], np.ndarray) or isinstance(sample["y"], tuple)
            ):  # for kcat_km, y is should have one value
                print("Skipped sample because y is multi value")
                return True
        elif self.args.enzyme_property == "km_value":
            raise NotImplementedError
        elif self.args.enzyme_property == "ph_optimum":
            raise NotImplementedError
        elif self.args.enzyme_property == "specific_activity":
            raise NotImplementedError
        elif self.args.enzyme_property == "temperature_optimum":
            raise NotImplementedError
        elif self.args.enzyme_property == "isoelectric_point":
            raise NotImplementedError
        elif self.args.enzyme_property == "ki_value":
            raise NotImplementedError
        elif self.args.enzyme_property == "ic50":
            raise NotImplementedError
        elif self.args.enzyme_property == "kcat_km":
            if not self.args.use_mean_labels and (
                isinstance(sample["y"], np.ndarray) or isinstance(sample["y"], tuple)
            ):  # for kcat_km, y is should have one value
                print("Skipped sample because y is multi value")
                return True
        elif self.args.enzyme_property == "ph_stability":
            raise NotImplementedError
        elif self.args.enzyme_property == "temperature_stability":
            raise NotImplementedError
        elif self.args.enzyme_property == "ph_range":
            raise NotImplementedError
        elif self.args.enzyme_property == "temperature_range":
            raise NotImplementedError
        elif self.args.enzyme_property == "localization":
            raise NotImplementedError
        elif self.args.enzyme_property == "tissue":
            raise NotImplementedError

        # check contradictory values TODO
        if not self.args.use_mean_labels:
            smi = sample["smiles"]
            seq = sample["sequence"]
            if any(
                not np.array_equal(i, sequence_smiles2y[f"{seq}{smi}"][0])
                if isinstance(i, np.ndarray)
                else i != sequence_smiles2y[f"{seq}{smi}"][0]
                for i in sequence_smiles2y[f"{seq}{smi}"]
            ):
                print("Skipped sample because of contradictory values")
                return True

        # check right split
        if hasattr(self, "to_split"):
            if self.args.split_type == "sequence":
                if self.to_split[sample["protein_id"]] != split_group:
                    return True

            if self.args.split_type == "ec":
                if self.to_split[sample["ec"]] != split_group:
                    return True

            if self.args.split_type == "random":
                seq = sample["sequence"]
                smi = sample["smiles"]
                if self.to_split[f"{seq}{smi}"] != split_group:
                    return True

        return False

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
    def set_args(args) -> None:
        super(BrendaConstants, BrendaConstants).set_args(args)
        if args.enzyme_property == "turnover_number":
            args.num_classes = 1
        elif args.enzyme_property == "km_value":
            raise NotImplementedError
        elif args.enzyme_property == "ph_optimum":
            raise NotImplementedError
        elif args.enzyme_property == "specific_activity":
            raise NotImplementedError
        elif args.enzyme_property == "temperature_optimum":
            raise NotImplementedError
        elif args.enzyme_property == "isoelectric_point":
            raise NotImplementedError
        elif args.enzyme_property == "ki_value":
            raise NotImplementedError
        elif args.enzyme_property == "ic50":
            raise NotImplementedError
        elif args.enzyme_property == "kcat_km":
            args.num_classes = 1
        elif args.enzyme_property == "ph_stability":
            raise NotImplementedError
        elif args.enzyme_property == "temperature_stability":
            raise NotImplementedError
        elif args.enzyme_property == "ph_range":
            raise NotImplementedError
        elif args.enzyme_property == "temperature_range":
            raise NotImplementedError
        elif args.enzyme_property == "localization":
            raise NotImplementedError
        elif args.enzyme_property == "tissue":
            raise NotImplementedError

    def get_label(self, value, property_name):
        # TODO - can values be 0?
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

    @property
    def SUMMARY_STATEMENT(self) -> None:
        """
        Prints summary statement with dataset stats
        """
        summary = f"\n{self.split_group} dataset for {self.args.enzyme_property} property contains {len(self.dataset)} samples"
        return summary

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        super(BrendaConstants, BrendaConstants).add_args(parser)
        parser.add_argument(
            "--use_mean_labels",
            action="store_true",
            default=False,
            help="If labels have more than one value, or multiple samples have different labels, use the mean",
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


@register_object("brenda_ec", "dataset")
class BrendaEC(Brenda):
    def load_dataset(self, args: argparse.ArgumentParser) -> None:
        super().load_dataset(args)
        ecs = list(self.metadata_json.keys())
        ecs = sorted(
            list(set([".".join(e.split(".")[: self.args.ec_level + 1]) for e in ecs]))
        )
        self.ec2class = {ec: i for i, ec in enumerate(ecs)}
        args.num_classes = len(ecs)

    def create_dataset(
        self, split_group: Literal["train", "dev", "test"]
    ) -> List[dict]:
        # map uniprot to EC number (or sub-EC number)
        uniprot2ec = defaultdict(list)
        for ec, ec_dict in tqdm(
            self.metadata_json.items(), desc="Iterating over Brenda"
        ):
            if not ec_dict.get("proteins", False):
                continue

            ec_task = ".".join(ec.split(".")[: self.args.ec_level + 1])

            for k, v in ec_dict["proteins"].items():
                for pid in v[0]["accessions"]:
                    uniprot2ec[pid].append(self.ec2class[ec_task])

        # create dataset of (protein, multi-task label) pairs
        dataset = []
        for protein_id, ec_list in tqdm(uniprot2ec.items(), desc="Creating dataset"):

            sample = {
                "sequence": self.brenda_proteins[protein_id]["sequence"],
                "protein_id": protein_id,
                "y": self.get_label(ec_list),
                "sample_id": f"prot{protein_id}",
            }

            if self.skip_sample(sample, split_group):
                continue

            dataset.append(sample)

        return dataset

    def get_label(self, ec_list):
        y = torch.zeros(self.args.num_classes)
        for ec in ec_list:
            y[ec] = 1
        return y

    def skip_sample(self, sample, split_group) -> bool:
        # if sequence is unknown
        if sample["sequence"] is None:
            return True

        # check right split
        if hasattr(self, "to_split"):
            if self.to_split[sample["protein_id"]] != split_group:
                return True

        return False


@register_object("brenda_reaction", "dataset")
class BrendaReaction(Brenda):
    def create_dataset(
        self, split_group: Literal["train", "dev", "test"]
    ) -> List[dict]:

        mcsa_data = self.load_mcsa_data(self.args)

        uniprot2reactions = defaultdict(list)

        # add brenda reactions
        for ec, ec_dict in tqdm(self.metadata_json.items(), desc="Creating dataset"):
            if "proteins" not in ec_dict:
                continue

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
                                    uniprotid = proteinid2uniprot[protein]
                                    sequence = self.brenda_proteins[uniprotid][
                                        "sequence"
                                    ]
                                    catalogued_reactions = [
                                        rxn["reaction_string"]
                                        for rxn in uniprot2reactions[uniprotid]
                                    ]
                                    if reaction_string not in catalogued_reactions:

                                        sample_id = hashlib.md5(
                                            f"{uniprotid}_{reaction_string}".encode()
                                        ).hexdigest()

                                        residues = self.get_uniprot_residues(
                                            mcsa_data, sequence, ec
                                        )

                                        sample = {
                                            "protein_id": uniprotid,
                                            "sequence": sequence,
                                            "reactants": rs,
                                            "products": ps,
                                            "ec": ec,
                                            "organism": protein2organism[protein],
                                            "reaction_string": ".".join(rs)
                                            + ">>"
                                            + ".".join(ps),
                                            "sample_id": sample_id,
                                            "residues": residues["residues"],
                                            "residue_mask": residues["residue_mask"],
                                            "has_residues": residues["has_residues"],
                                            "residue_positions": residues[
                                                "residue_positions"
                                            ],
                                        }

                                        if self.skip_sample(sample, split_group):
                                            continue

                                        sample["reactants"] = [
                                            self.get_smiles(m)
                                            for m in sample["reactants"]
                                        ]

                                        sample["products"] = [
                                            self.get_smiles(m)
                                            for m in sample["products"]
                                        ]

                                        uniprot2reactions[uniprotid].append(sample)

        # add M-CSA data not in brenda
        for sequence, uniprot_dict in tqdm(
            mcsa_data.items(), desc="Adding M-CSA reactions"
        ):
            uniprotid = uniprot_dict["uniprot"]
            if uniprotid in uniprot2reactions:
                continue

            for ec, ec_dict in uniprot_dict.items():

                if ec in ["sequence", "uniprot"]:
                    continue

                if any(
                    s in [None, []] for s in ec_dict["reactants"] + ec_dict["products"]
                ):
                    continue

                residues = self.get_uniprot_residues(mcsa_data, sequence, ec)

                rs = sorted(ec_dict["reactants"])
                ps = sorted(ec_dict["products"])

                reaction_string = ".".join(rs) + ">>" + ".".join(ps)

                sample_id = hashlib.md5(
                    f"{uniprotid}_{reaction_string}".encode()
                ).hexdigest()

                sample = {
                    "protein_id": uniprotid,
                    "sequence": uniprot_dict["sequence"],
                    "reactants": rs,
                    "products": ps,
                    "ec": ec,
                    "reaction_string": reaction_string,
                    "sample_id": sample_id,
                    "residues": residues["residues"],
                    # "residue_mask": residues["residue_mask"],
                    "residue_positions": residues["residue_positions"],
                    "has_residues": True,
                }

                if self.skip_sample(sample, split_group):
                    continue

                uniprot2reactions[uniprotid].append(sample)

        # make each reaction a sample
        all_reactions = set()
        dataset = []
        for uniprot, reaction_list in uniprot2reactions.items():
            for reaction in reaction_list:

                if self.skip_sample(reaction, split_group):
                    continue

                # in case using reactions alone without protein information in model
                if self.args.deduplicate_reactions:
                    rxn = "{}>>{}".format(
                        ".".join(reaction["reactants"]), ".".join(reaction["products"])
                    )
                    if rxn in all_reactions:
                        continue

                    all_reactions.add(rxn)

                dataset.append(reaction)

        return dataset

    def skip_sample(self, sample, split_group) -> bool:
        # check if sample has mol
        if "?" in (sample["products"] + sample["reactants"]):
            return True

        if any(
            s in [None, [], "?"] for s in (sample["products"] + sample["reactants"])
        ):
            return True

        # if sequence is unknown
        if sample["sequence"] is None:
            return True

        # check right split
        if hasattr(self, "to_split"):
            if self.args.split_type == "sequence":
                if self.to_split[sample["protein_id"]] != split_group:
                    return True

            if self.args.split_type == "ec":
                ec = ".".join(sample["ec"].split(".")[: self.args.ec_level + 1])
                if self.to_split[ec] != split_group:
                    return True
        return False

    def get_uniprot_residues(self, mcsa_data, sequence, ec):
        """Get residues from MCSA data

        Args:
            mcsa_data (dict): MCSA data
            sequence (str): protein sequence
            ec (str): ec number

        Returns:
            dict: {residue_mask: torch.Tensor, has_residues: torch.Tensor, residues: [smiles]}
        """
        if sequence is None:
            return {
                "residue_mask": torch.zeros(1),
                "has_residues": 0,
                "residues": [],
                "residue_positions": [],
            }

        y = torch.zeros(len(sequence))
        has_y = 0
        residues = []
        residue_pos = []
        if mcsa_data.get(sequence, False):
            if mcsa_data[sequence].get(ec, False):
                for residue_dict in mcsa_data[sequence][ec]["residues"]:
                    if (residue_dict["residue_id"] is None) or (
                        residue_dict["residue"] == ""
                    ):
                        continue
                    letter = sequence[residue_dict["residue_id"]]
                    amino_acid = AA_TO_SMILES.get(letter, None)  # consider skipping
                    y[residue_dict["residue_id"]] = 1
                    residue_pos.append(residue_dict["residue_id"])
                    residues.append(amino_acid)
                has_y = 1

        return {
            "residue_mask": y,
            "has_residues": has_y,
            "residues": residues,
            "residue_positions": residue_pos,
        }

    def __getitem__(self, index):
        sample = self.dataset[index]

        try:
            reactants, products = copy.deepcopy(sample["reactants"]), copy.deepcopy(
                sample["products"]
            )

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
                "organism": sample.get("organism", "none"),
                "protein_id": sample["protein_id"],
                "sample_id": sample["sample_id"],
                "residues": ".".join(sample["residues"]),
                "residue_mask": sample["residue_mask"],
                "has_residues": sample["has_residues"],
                "residue_positions": ".".join(
                    [str(s) for s in sample["residue_positions"]]
                ),
            }

            return item

        except Exception as e:
            warnings.warn(
                f"Could not load sample: {sample['sample_id']} because of exception: {e}"
            )

    @property
    def SUMMARY_STATEMENT(self) -> None:
        reactions = [
            "{}>>{}".format(".".join(d["reactants"]), ".".join(d["products"]))
            for d in self.dataset
        ]
        proteins = [d["protein_id"] for d in self.dataset]
        ecs = [d["ec"] for d in self.dataset]
        statement = f""" 
        * Number of reactions: {len(set(reactions))}
        * Number of proteins: {len(set(proteins))}
        * Number of ECs: {len(set(ecs))}
        """
        return statement


@register_object("mcsa", "dataset")
class MCSA(BrendaReaction):
    def create_dataset(
        self, split_group: Literal["train", "dev", "test"]
    ) -> List[dict]:

        mcsa_data = self.load_mcsa_data(self.args)

        uniprot2reactions = defaultdict(list)
        for sequence, uniprot_dict in tqdm(
            mcsa_data.items(), desc="Making M-CSA dataset"
        ):

            uniprotid = uniprot_dict["uniprot"]

            for ec, ec_dict in uniprot_dict.items():

                if ec in ["sequence", "uniprot"]:
                    continue

                residues = self.get_uniprot_residues(mcsa_data, sequence, ec)
                rs = ec_dict["reactants"]
                ps = ec_dict["products"]

                uniprot2reactions[uniprotid].append(
                    {
                        "protein_id": uniprotid,
                        "sequence": sequence,
                        "reactants": rs,
                        "products": ps,
                        "ec": ec,
                        "residues": residues["residues"],
                        "residue_mask": residues["residue_mask"],
                        "has_residues": residues["has_residues"],
                        "residue_positions": residues["residue_positions"],
                    }
                )
        # make each reaction a sample
        dataset = []
        for uniprot, reaction_list in uniprot2reactions.items():
            for reaction in reaction_list:
                if self.skip_sample(reaction, split_group):
                    continue
                reaction_string = (
                    ".".join(reaction["reactants"])
                    + ">>"
                    + ".".join(reaction["products"])
                )
                reaction["reaction_string"] = reaction_string
                reaction["sample_id"] = hashlib.md5(
                    f"{uniprotid}_{reaction_string}".encode()
                ).hexdigest()
                dataset.append(reaction)

        return dataset

    def skip_sample(self, sample, split_group) -> bool:
        # check if sample has mol
        if self.args.mcsa_skip_unk_smiles:
            if "?" in (sample["products"] + sample["reactants"]):
                return True

            if any(s in [None, []] for s in sample["reactants"] + sample["products"]):
                return True

        # if sequence is unknown
        if sample["sequence"] is None:
            return True

        if (self.args.max_protein_length is not None) and len(
            sample["sequence"]
        ) > self.args.max_protein_length:
            return True

        # check right split
        if hasattr(self, "to_split"):
            if self.args.split_type == "sequence":
                if self.to_split[sample["protein_id"]] != split_group:
                    return True

            if self.args.split_type == "ec":
                ec = ".".join(sample["ec"].split(".")[: self.args.ec_level + 1])
                if self.to_split[ec] != split_group:
                    return True

        return False
