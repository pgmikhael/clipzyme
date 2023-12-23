from typing import List, Literal
from nox.utils.registry import register_object, get_object
from nox.datasets.abstract import AbstractDataset
from tqdm import tqdm
import argparse
import pickle
import warnings
import copy, os
import numpy as np
import random
from random import Random
from collections import defaultdict, Counter
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem as rdk
import torch
from torch_geometric.data import HeteroData, Data, Batch, Dataset
import hashlib
import json
from Bio.Data.IUPACData import protein_letters_3to1
from esm import pretrained
import Bio
import Bio.PDB
from rxn.chemutils.smiles_randomization import randomize_smiles_rotated

from nox.utils.smiles import (
    get_rdkit_feature,
    remove_atom_maps_manual,
    assign_dummy_atom_maps,
    generate_scaffold,
)
from nox.utils.protein_utils import (
    read_structure_file,
    filter_resolution,
    build_graph,
    compute_graph_edges,
    precompute_node_embeddings,
    compute_node_embedding,
    get_sequences,
)
from nox.utils.pyg import from_smiles, from_mapped_smiles
from nox.utils.wln_processing import get_bond_changes
from nox.utils.amino_acids import AA_TO_SMILES
from nox.models.wln import WLDN_Cache

ESM_MODEL2HIDDEN_DIM = {
    "esm2_t48_15B_UR50D": 5120,
    "esm2_t36_3B_UR50D": 2560,
    "esm2_t33_650M_UR50D": 1280,
    "esm2_t30_150M_UR50D": 640,
    "esm2_t12_35M_UR50D": 480,
    "esm2_t6_8M_UR50D": 320,
}


def stringify_sets(sets):
    final = []
    for i in sets:
        s = str(i[0]) + "-" + str(i[1]) + "-" + str(i[2])
        final.append(s)
    return " ".join(final)


def destringify_sets(x: str):
    return [
        (int(l.split("-")[0]), int(l.split("-")[1]), float(l.split("-")[2]))
        for l in x.split(" ")
    ]


@register_object("enzymemap_reactions", "dataset")
class EnzymeMap(AbstractDataset):
    def __init__(self, args, split_group) -> None:
        if args.use_protein_graphs:
            self.esm_dir = args.esm_dir
            model, alphabet = pretrained.load_model_and_alphabet(args.esm_dir)
            self.esm_model = model
            self.alphabet = alphabet
            self.batch_converter = alphabet.get_batch_converter()
        super(EnzymeMap, EnzymeMap).__init__(self, args, split_group)
        self.metadata_json = None  # overwrite for memory
        if args.load_wln_cache_in_dataset:
            self.cache = WLDN_Cache(args.cache_path)

    def init_class(self, args: argparse.ArgumentParser, split_group: str) -> None:
        """Perform Class-Specific init methods
           Default is to load JSON dataset

        Args:
            args (argparse.ArgumentParser)
            split_group (str)
        """
        self.version = args.version
        self.load_dataset(args)

        self.valid_ec2uniprot = defaultdict(set)

        self.ec2uniprot = pickle.load(
            open(
                "/Mounts/rbg-storage1/datasets/Enzymes/EnzymeMap/ec2uniprot.p",
                "rb",
            )
        )
        self.uniprot2sequence = pickle.load(
            open(
                "/Mounts/rbg-storage1/datasets/Enzymes/EnzymeMap/uniprot2sequence.p",
                "rb",
            )
        )
        self.uniprot2sequence_len = {
            k: 0 if v is None else len(v) for k, v in self.uniprot2sequence.items()
        }
        self.uniprot2cluster = pickle.load(
            # open(
            #     "/Mounts/rbg-storage1/datasets/Enzymes/EnzymeMap/mmseq_clusters.p",  # TODO
            #     "rb",
            # )
            open(
                args.uniprot2cluster_path,
                "rb",
            )
        )

        # products to remove based on smiles or pattern
        remove_patterns_path = (
            "/Mounts/rbg-storage1/datasets/Enzymes/ECReact/patterns.txt"
        )
        remove_molecules_path = (
            "/Mounts/rbg-storage1/datasets/Enzymes/ECReact/molecules.txt"
        )

        self.remove_patterns = []
        self.remove_molecules = []

        for line in open(remove_patterns_path):
            if not line.startswith("//") and line.strip():
                self.remove_patterns.append(line.split("//")[0].strip())

        self.remove_patterns = [
            rdk.MolFromSmarts(smart_pattern) for smart_pattern in self.remove_patterns
        ]

        for line in open(remove_molecules_path):
            if not line.startswith("//") and line.strip():
                smiles = line.split("//")[0].strip()
                mol = rdk.MolFromSmiles(smiles)
                if mol:
                    self.remove_molecules.append(rdk.MolToSmiles(mol))
                    self.remove_molecules.append(
                        Chem.CanonSmiles(
                            self.remove_molecules[-1].replace("[O-]", "[OH]")
                        )
                    )

        if (
            not hasattr(self.args, "use_all_sequences")
            or not self.args.use_all_sequences
        ):
            self.uniprot2split = pickle.load(
                open(
                    "/Mounts/rbg-storage1/datasets/Enzymes/EnzymeMap/mmseq_splits_precomputed.p",
                    "rb",
                )
            )

    def create_dataset(
        self, split_group: Literal["train", "dev", "test"]
    ) -> List[dict]:
        # if removing top K
        if self.args.topk_byproducts_to_remove is not None:
            raw_byproducts = Counter(
                [r for d in self.metadata_json for r in d["products"]]
            ).most_common(self.args.topk_byproducts_to_remove)
            mapped_byproducts = Counter(
                [r for d in self.metadata_json for r in d.get("mapped_products", [])]
            ).most_common(self.args.topk_byproducts_to_remove)
            self.common_byproducts = {
                s[0]: True
                for byproducts in [raw_byproducts, mapped_byproducts]
                for s in byproducts
            }

        dataset = []

        for rowid, reaction in tqdm(
            enumerate(self.metadata_json),
            desc="Building dataset",
            total=len(self.metadata_json),
            ncols=100,
        ):
            self.mol2size = {}

            ec = reaction["ec"]
            reactants = (
                sorted(reaction.get("mapped_reactants", []))
                if self.args.use_mapped_reaction
                else sorted(reaction["reactants"])
            )
            products = (
                sorted(reaction.get("mapped_products", []))
                if self.args.use_mapped_reaction
                else sorted(reaction["products"])
            )
            products = [p for p in products if p not in reactants]

            if self.args.topk_byproducts_to_remove is not None:
                products = [p for p in products if p not in self.common_byproducts]

            # select uniprots
            if self.args.version == "1":
                alluniprots = self.ec2uniprot.get(ec, [])
                protein_refs = []
            elif self.args.version == "2":
                protein_refs = eval(reaction["protein_refs"])
                alluniprots = protein_refs
                if (len(alluniprots) == 0) and self.args.sample_uniprot_per_ec:
                    alluniprots = self.ec2uniprot.get(ec, [])

            valid_uniprots = []
            for uniprot in alluniprots:
                temp_sample = {
                    "quality": reaction["quality"],
                    "reactants": reactants,
                    "products": products,
                    "ec": ec,
                    "protein_id": uniprot,
                    "protein_db": reaction.get("protein_db", ""),
                    "protein_refs": protein_refs,
                    "organism": reaction.get("organism", ""),
                    "rule_id": reaction["rule_id"],
                }
                if self.skip_sample(temp_sample, split_group):
                    continue

                valid_uniprots.append(uniprot)

            if len(valid_uniprots) == 0:
                continue

            for uniprot in valid_uniprots:
                sample = {
                    "quality": reaction["quality"],
                    "reactants": reactants,
                    "products": products,
                    "ec": ec,
                    "rowid": f"{uniprot}_{reaction['rxnid']}",
                    "uniprot_id": uniprot,
                    "protein_id": uniprot,
                    "organism": reaction.get("organism", ""),
                    "rule_id": reaction["rule_id"],
                }
                if "split" in reaction:
                    sample["split"] = reaction["split"]
                # add reaction sample to dataset
                dataset.append(sample)

                for ec_level, _ in enumerate(ec.split(".")):
                    sample[f"ec{ec_level+1}"] = ".".join(
                        ec.split(".")[: (ec_level + 1)]
                    )

        return dataset

    def skip_sample(self, sample, split_group) -> bool:
        if sample["quality"] < self.args.min_reaction_quality:
            return True

        if "-" in sample["ec"]:
            return True

        # if sequence is unknown
        sequence = self.uniprot2sequence.get(sample["protein_id"], None)
        if (sequence is None) or (len(sequence) == 0):
            return True

        if (self.args.max_protein_length is not None) and len(
            sequence
        ) > self.args.max_protein_length:
            return True

        if self.args.max_reactant_size is not None:
            for mol in sample["reactants"]:
                if not (mol in self.mol2size):
                    self.mol2size[mol] = rdkit.Chem.MolFromSmiles(mol).GetNumAtoms()

                if self.mol2size[mol] > self.args.max_reactant_size:
                    return True

        if self.args.max_product_size is not None:
            for mol in sample["products"]:
                if not (mol in self.mol2size):
                    self.mol2size[mol] = rdkit.Chem.MolFromSmiles(mol).GetNumAtoms()

                if self.mol2size[mol] > self.args.max_product_size:
                    return True
            # if self.mol2size[mol] < 2:
            #     return True

        if len(sample["products"]) > self.args.max_num_products:
            return True

        if ("bond_changes" in sample) and (len(sample["bond_changes"]) == 0):
            return True

        if (self.version == "2") and (
            sample["protein_db"] not in ["swissprot", "uniprot"]
        ):
            if len(sample["protein_refs"]) > 0:  # ids obtained from reference not used
                return True

            if len(sample["protein_refs"]) == 0 and (
                not self.args.sample_uniprot_per_ec
            ):
                return True

        if self.args.remove_duplicate_reactions:
            # reaction = "{}>>{}".format(
            #     remove_atom_maps_manual(".".join(sample["reactants"])),
            #     remove_atom_maps_manual(".".join(sample["products"])),
            # )

            reaction = "{}|{}".format(sample["reaction_string"], sample["uniprot_id"])
            if reaction in self.unique_reactions:
                return True
            self.unique_reactions.add(reaction)

        return False

    def __getitem__(self, index):
        sample = self.dataset[index]

        try:
            reactants, products = copy.deepcopy(sample["reactants"]), copy.deepcopy(
                sample["products"]
            )

            ec = sample["ec"]
            uniprot_id = sample.get("uniprot_id", "unk")
            sequence = self.uniprot2sequence.get(uniprot_id, "<unk>")

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

            # remove atom-mapping if applicable
            reactants = ".".join(reactants)
            products = ".".join(products)
            # reactants = remove_atom_maps(reactants)
            # products = remove_atom_maps(products)

            # remove stereochemistry
            if self.args.remove_stereochemistry:
                reactants_mol = Chem.MolFromSmiles(reactants)
                products_mol = Chem.MolFromSmiles(products)
                Chem.RemoveStereochemistry(reactants_mol)
                Chem.RemoveStereochemistry(products_mol)
                reactants = Chem.MolToSmiles(reactants_mol)
                products = Chem.MolToSmiles(products_mol)

            sample_id = sample["rowid"]
            item = {
                "reaction": reaction,
                "reactants": reactants,
                "products": products,
                "sequence": sequence,
                "ec": ec,
                "organism": sample.get("organism", "none"),
                "protein_id": uniprot_id,
                "sample_id": sample_id,
                "smiles": ".".join(products),
                "all_smiles": list(
                    self.reaction_to_products[
                        f"{ec}{'.'.join(sorted(sample['reactants']))}"
                    ]
                ),
                "quality": sample["quality"],
            }

            split_ec = ec.split(".")
            for k, v in self.args.ec_levels.items():
                item[f"ec{k}"] = v.get(".".join(split_ec[: int(k)]), -1)

            if self.args.use_pesto_scores:
                scores = self.get_pesto_scores(item["protein_id"])
                if (scores is None) or (scores.shape[0] != len(item["sequence"])):
                    # make all zeros of length sequence
                    scores = torch.zeros(len(item["sequence"]))
                item["sequence_annotation"] = scores

            return item

        except Exception as e:
            print(
                f"Getitem enzymemap: Could not load sample {sample['uniprot_id']} because of exception {e}"
            )

    def get_pesto_scores(self, uniprot):
        filepath = f"{self.args.pesto_scores_directory}/AF-{uniprot}-F1-model_v4.pt"
        if not os.path.exists(filepath):
            return None
        scores_dict = torch.load(filepath)
        chain = "A:0"  # * NOTE: hardcoded because currently only option
        residue_ids = scores_dict[chain]["resid"]
        residue_ids_unique = np.unique(residue_ids, return_index=True)[1]
        scores = scores_dict[chain]["ligand"][residue_ids_unique]
        return torch.tensor(scores)

    def assign_splits(self, metadata_json, split_probs, seed=0) -> None:
        """
        Assigns each sample to a split group based on split_probs
        """
        # get all samples
        self.to_split = {}

        # set seed
        np.random.seed(seed)

        # assign groups
        if self.args.split_type in [
            "mmseqs",
            "sequence",
            "ec",
            "product",
            "mmseqs_precomputed",
        ]:
            if (
                self.args.split_type == "mmseqs"
                or self.args.split_type == "mmseqs_precomputed"
            ):
                samples = [
                    self.uniprot2cluster[reaction["uniprot_id"]]
                    for reaction in metadata_json
                ]
                # samples = list(self.uniprot2cluster.values())

            if self.args.split_type == "sequence":
                # split based on uniprot_id
                samples = [
                    u
                    for reaction in metadata_json
                    for u in self.ec2uniprot.get(reaction["ec"], [])
                ]
                if "protein_id" in metadata_json[0]:
                    samples += [r["protein_id"] for r in metadata_json]

            elif self.args.split_type == "ec":
                # split based on ec number
                samples = [reaction["ec"] for reaction in metadata_json]

                # option to change level of ec categorization based on which to split
                samples = [
                    ".".join(e.split(".")[: self.args.ec_level + 1]) for e in samples
                ]

            elif self.args.split_type == "product":
                # split by reaction product (splits share no products)
                samples = [".".join(s["products"]) for s in metadata_json]

            sample2count = Counter(samples)
            samples = sorted(list(set(samples)))
            np.random.shuffle(samples)
            samples_cumsum = np.cumsum([sample2count[s] for s in samples])
            # Find the indices for each quantile
            split_indices = [
                np.searchsorted(
                    samples_cumsum, np.round(q, 3) * samples_cumsum[-1], side="right"
                )
                for q in np.cumsum(split_probs)
            ]
            split_indices[-1] = len(samples)
            split_indices = np.concatenate([[0], split_indices])
            for i in range(len(split_indices) - 1):
                self.to_split.update(
                    {
                        sample: ["train", "dev", "test"][i]
                        for sample in samples[split_indices[i] : split_indices[i + 1]]
                    }
                )

            # split_indices = np.ceil(
            #     np.cumsum(np.array(split_probs) * len(samples))
            # ).astype(int)
            # split_indices = np.concatenate([[0], split_indices])

            # for i in range(len(split_indices) - 1):
            #     self.to_split.update(
            #         {
            #             sample: ["train", "dev", "test"][i]
            #             for sample in samples[split_indices[i] : split_indices[i + 1]]
            #         }
            #     )

        elif self.args.split_type == "rule_id":
            # rule id
            rules = [reaction["rule_id"] for reaction in metadata_json]
            rule2count = Counter(rules)
            samples = sorted(list(set(rules)))
            np.random.shuffle(samples)
            samples_cumsum = np.cumsum([rule2count[s] for s in samples])
            # Find the indices for each quantile
            split_indices = [
                np.searchsorted(
                    samples_cumsum, np.round(q, 3) * samples_cumsum[-1], side="right"
                )
                for q in np.cumsum(split_probs)
            ]
            split_indices[-1] = len(samples)
            split_indices = np.concatenate([[0], split_indices])
            for i in range(len(split_indices) - 1):
                self.to_split.update(
                    {
                        sample: ["train", "dev", "test"][i]
                        for sample in samples[split_indices[i] : split_indices[i + 1]]
                    }
                )

        elif self.args.split_type == "ec_hold_out":
            unique_products = set(
                [
                    ".".join(sample["products"])
                    for sample in self.metadata_json
                    if sample["ec"].split(".")[0] != str(self.args.held_out_ec_num)
                ]
            )
            # ! ENSURE REPRODUCIBLE SETS FOR SAME SEED
            unique_products = sorted(list(unique_products))
            np.random.shuffle(unique_products)

            dev_probs = split_probs[1] / (split_probs[0] + split_probs[1])
            train_probs = split_probs[0] / (split_probs[0] + split_probs[1])
            if not self.args.split_multiproduct_samples:
                products2split = {
                    p: np.random.choice(["train", "dev"], p=[train_probs, dev_probs])
                    for p in unique_products
                }
            else:
                products2split = {}
                for p_list in unique_products:
                    for p in p_list.split("."):
                        products2split[p] = np.random.choice(
                            ["train", "dev"], p=[train_probs, dev_probs]
                        )

            for sample in self.metadata_json:
                ec = sample["ec"]
                rkey = (
                    "mapped_reactants"
                    if "mapped_reactants" in self.metadata_json[0]
                    else "reactants"
                )
                pkey = (
                    "mapped_products"
                    if "mapped_products" in self.metadata_json[0]
                    else "products"
                )
                reactants = sorted([s for s in sample[rkey] if s != "[H+]"])
                products = sorted([s for s in sample[pkey] if s != "[H+]"])
                products = [p for p in products if p not in reactants]

                if self.args.topk_byproducts_to_remove is not None:
                    products = [p for p in products if p not in self.common_byproducts]

                reaction_string = "{}>>{}".format(
                    ".".join(reactants), ".".join(products)
                )

                if self.args.version == "1":
                    alluniprots = self.ec2uniprot.get(ec, [])
                    protein_refs = []
                elif self.args.version == "2":
                    protein_refs = eval(sample["protein_refs"])
                    alluniprots = protein_refs
                    if (len(alluniprots) == 0) and self.args.sample_uniprot_per_ec:
                        alluniprots = self.ec2uniprot.get(ec, [])

                if (
                    self.args.create_sample_per_sequence
                    or self.args.sample_uniprot_per_ec
                ):
                    valid_uniprots = []
                    for uniprot in alluniprots:
                        if self.args.split_multiproduct_samples:
                            for product_id, p in enumerate(products):
                                psample = copy.deepcopy(sample)
                                psample["products"] = [p]
                                # psample["sample_id"] += f"_{product_id}"
                                preaction_string = "{}>>{}".format(
                                    ".".join(psample["reactants"]), p
                                )
                                # uniprot = psample["uniprot_id"]
                                punique_sample_content = f"{preaction_string}{uniprot}{psample.get('organism', '')}"
                                phashed_sample_content = hashlib.sha256(
                                    punique_sample_content.encode("utf-8")
                                ).hexdigest()
                                psample["hash_sample_id"] = phashed_sample_content
                                if str(self.args.held_out_ec_num) == ec:
                                    self.to_split[psample["hash_sample_id"]] = "test"
                                else:
                                    self.to_split[
                                        psample["hash_sample_id"]
                                    ] = products2split[p]

                        else:
                            unique_sample_content = f"{reaction_string}{uniprot}{sample.get('organism', '')}"
                            hashed_sample_content = hashlib.sha256(
                                unique_sample_content.encode("utf-8")
                            ).hexdigest()
                            sample["hash_sample_id"] = hashed_sample_content
                            if sample["ec"].split(".")[0] == str(
                                self.args.held_out_ec_num
                            ):
                                self.to_split[sample["hash_sample_id"]] = "test"
                            else:
                                self.to_split[
                                    sample["hash_sample_id"]
                                ] = products2split[".".join(sample["products"])]

        # random splitting
        elif self.args.split_type == "random":
            for sample in self.metadata_json:
                reaction_string = (
                    ".".join(sample["reactants"]) + ">>" + ".".join(sample["products"])
                )
                self.to_split.update(
                    {
                        reaction_string: np.random.choice(
                            ["train", "dev", "test"], p=split_probs
                        )
                    }
                )
        elif self.args.split_type == "scaffold":
            # split based on scaffold
            self.scaffold_split(metadata_json, split_probs, seed)
        else:
            raise ValueError("Split type not supported")

    def scaffold_split(self, meta: List[dict], split_probs: List[float], seed):
        scaffold_to_indices = defaultdict(list)
        for m_i, m in enumerate(meta):
            scaffold = generate_scaffold(m["smiles"])
            scaffold_to_indices[scaffold].append(m_i)

        # Split
        train_size, val_size, test_size = (
            split_probs[0] * len(meta),
            split_probs[1] * len(meta),
            split_probs[2] * len(meta),
        )
        train, val, test = [], [], []
        train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

        # Seed randomness
        random = Random(seed)

        if (
            self.args.scaffold_balanced
        ):  # Put stuff that's bigger than half the val/test size into train, rest just order randomly
            index_sets = list(scaffold_to_indices.values())
            big_index_sets = []
            small_index_sets = []
            for index_set in index_sets:
                if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
                    big_index_sets.append(index_set)
                else:
                    small_index_sets.append(index_set)
            random.seed(seed)
            random.shuffle(big_index_sets)
            random.shuffle(small_index_sets)
            index_sets = big_index_sets + small_index_sets
        else:  # Sort from largest to smallest scaffold sets
            index_sets = sorted(
                list(scaffold_to_indices.values()),
                key=lambda index_set: len(index_set),
                reverse=True,
            )

        for index_set in index_sets:
            if len(train) + len(index_set) <= train_size:
                train += index_set
                train_scaffold_count += 1
            elif len(val) + len(index_set) <= val_size:
                val += index_set
                val_scaffold_count += 1
            else:
                test += index_set
                test_scaffold_count += 1

        for idx_list, split in [(train, "train"), (val, "dev"), (test, "test")]:
            for idx in idx_list:
                meta[idx]["split"] = split
                if (
                    meta[idx]["smiles"] in self.to_split
                    and self.to_split[meta[idx]["smiles"]] != split
                ):
                    raise Exception("Smile exists in to_split but with different split")
                self.to_split[meta[idx]["smiles"]] = split

    def get_split_group_dataset(
        self, processed_dataset, split_group: Literal["train", "dev", "test"]
    ) -> List[dict]:
        dataset = []
        for sample in processed_dataset:
            # check right split
            if self.args.split_type == "ec":
                split_ec = sample[f"ec{self.args.ec_level + 1}"]
                if self.to_split[split_ec] != split_group:
                    continue

            elif self.args.split_type == "rule_id":
                if self.to_split[sample["rule_id"]] != split_group:
                    continue

            elif self.args.split_type == "mmseqs":
                cluster = self.uniprot2cluster.get(sample["protein_id"], None)
                if (cluster is None) or (self.to_split[cluster] != split_group):
                    continue
            elif (
                self.args.split_type == "mmseqs_precomputed"
                or self.args.split_type == "scaffold"
            ):
                if sample["split"] != split_group:
                    continue
            elif self.args.split_type in ["product"]:
                products = ".".join(sample["products"])
                if self.to_split[products] != split_group:
                    continue

            elif self.args.split_type == "sequence":
                uniprot = sample["protein_id"]
                if self.to_split[uniprot] != split_group:
                    continue

            elif self.args.split_type == "ec_hold_out":
                sample_id = sample["hash_sample_id"]
                if self.to_split[sample_id] != split_group:
                    continue

            elif sample["split"] is not None:
                if sample["split"] != split_group:
                    continue
            dataset.append(sample)
        return dataset

    def post_process(self, args):
        # add all possible products
        reaction_to_products = defaultdict(set)
        for sample in self.dataset:
            key = f"{sample['ec']}{'.'.join(sample['reactants'])}"
            reaction_to_products[key].add(".".join(sample["products"]))
        self.reaction_to_products = reaction_to_products

        # set ec levels to id for use in modeling
        ecs = [d["ec"].split(".") for d in self.dataset]
        args.ec_levels = {}
        for level in range(1, 5, 1):
            unique_classes = sorted(list(set(".".join(ec[:level]) for ec in ecs)))
            args.ec_levels[str(level)] = {c: i for i, c in enumerate(unique_classes)}

    def remove_from_products(self, product):
        mol = Chem.MolFromSmiles(product)
        for mol_pattern in self.remove_patterns:
            if mol.HasSubstructMatch(mol_pattern):
                return True
        if product in self.remove_molecules:
            return True
        return False

    def create_protein_graph(self, sample):
        try:
            raw_path = os.path.join(
                self.args.protein_structures_dir,
                f"AF-{sample['uniprot_id']}-F1-model_v4.cif",
            )
            protein_args = {
                "sample_id": sample["sample_id"],
                "protein_parser": Bio.PDB.MMCIFParser(),
                "protein_resolution": "residue",
                "graph_edge_args": {"knn_size": 10},
                "center_protein": True,
            }

            sample_id = protein_args["sample_id"]
            protein_parser = protein_args["protein_parser"]
            protein_resolution = protein_args["protein_resolution"]
            graph_edge_args = protein_args["graph_edge_args"]
            center_protein = protein_args["center_protein"]

            # parse pdb
            all_res, all_atom, all_pos = read_structure_file(
                protein_parser, raw_path, sample_id
            )
            # filter resolution of protein (backbone, atomic, etc.)
            atom_names, seq, pos = filter_resolution(
                all_res,
                all_atom,
                all_pos,
                protein_resolution=protein_resolution,
            )
            # generate graph
            data = build_graph(atom_names, seq, pos, sample_id)
            # kNN graph
            data = compute_graph_edges(data, **graph_edge_args)
            if center_protein:
                center = data["receptor"].pos.mean(dim=0, keepdim=True)
                data["receptor"].pos = data["receptor"].pos - center
                data.center = center
            uniprot_id = sample["uniprot_id"]
            sequence = self.uniprot2sequence[uniprot_id]
            data.structure_sequence = self.uniprot2sequence[uniprot_id]

            node_embeddings_args = {
                "model": self.esm_model,
                "model_location": self.esm_dir,
                "alphabet": self.alphabet,
                "batch_converter": self.batch_converter,
            }

            embedding_path = os.path.join(
                self.args.protein_graphs_dir,
                "precomputed_node_embeddings",
                f"{sample['uniprot_id']}.pt",
            )

            if os.path.exists(embedding_path):
                node_embedding = torch.load(sample["embedding_path"])
            else:
                node_embedding = compute_node_embedding(data, **node_embeddings_args)
            # Fix sequence length mismatches
            if len(data["receptor"].seq) != node_embedding.shape[0]:
                print("Computing seq embedding for mismatched seq length")
                protein_letters_3to1.update(
                    {k.upper(): v for k, v in protein_letters_3to1.items()}
                )
                AA_seq = ""
                for char in seq:
                    AA_seq += protein_letters_3to1[char]
                # sequences = get_sequences(
                #     self.protein_parser,
                #     [sample["sample_id"]],
                #     [os.path.join(self.structures_dir, f"AF-{sample['uniprot_id']}-F1-model_v4.cif")],
                # )

                data.structure_sequence = AA_seq
                data["receptor"].x = compute_node_embedding(
                    data, **node_embeddings_args
                )
            else:
                data["receptor"].x = node_embedding

            if len(data["receptor"].seq) != data["receptor"].x.shape[0]:
                return None

            return data

        except Exception as e:
            print(
                f"Create prot graph: Could not load sample {sample['uniprot_id']} because of the exception {e}"
            )
            return None

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        super(EnzymeMap, EnzymeMap).add_args(parser)
        parser.add_argument(
            "--held_out_ec_num",
            type=int,
            default=None,
            help="EC number to hold out",
        )
        parser.add_argument(
            "--uniprot2cluster_path",
            type=str,
            default="/Mounts/rbg-storage1/datasets/Enzymes/EnzymeMap/mmseq_clusters_updated.p",
            help="path to uniprot2cluster pickle",
        )
        parser.add_argument(
            "--esm_dir",
            type=str,
            default="/Mounts/rbg-storage1/snapshots/metabolomics/esm2/checkpoints/esm2_t33_650M_UR50D.pt",
            help="directory to load esm model from",
        )
        parser.add_argument(
            "--use_protein_graphs",
            action="store_true",
            default=False,
            help="whether to use and generate protein graphs",
        )
        parser.add_argument(
            "--protein_graphs_dir",
            type=str,
            default=None,
            help="directory to load protein graphs from",
        )
        parser.add_argument(
            "--protein_structures_dir",
            type=str,
            default=None,
            help="directory to load protein graphs from",
        )
        parser.add_argument(
            "--ec_level",
            type=int,
            default=3,
            choices=[0, 1, 2, 3],
            help="EC level to use (e.g., ec_level 1 of '1.2.3.1' -> '1.2')",
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
            "--max_protein_length",
            type=int,
            default=None,
            help="skip proteins longer than max_protein_length",
        )
        parser.add_argument(
            "--use_mapped_reaction",
            action="store_true",
            default=False,
            help="use atom-mapped reactions",
        )
        parser.add_argument(
            "--max_reactant_size",
            type=int,
            default=None,
            help="maximum reactant size",
        )
        parser.add_argument(
            "--max_product_size",
            type=int,
            default=None,
            help="maximum reactant size",
        )
        parser.add_argument(
            "--max_num_products",
            type=int,
            default=np.inf,
            help="maximum number of products",
        )
        parser.add_argument(
            "--topk_byproducts_to_remove",
            type=int,
            default=None,
            help="remove common byproducts",
        )
        parser.add_argument(
            "--use_pesto_scores",
            action="store_true",
            default=False,
            help="use pesto scores",
        )
        parser.add_argument(
            "--pesto_scores_directory",
            type=str,
            default="/Mounts/rbg-storage1/datasets/Enzymes/ECReact/pesto_ligands",
            help="load pesto scores from directory predictions",
        )
        parser.add_argument(
            "--create_sample_per_sequence",
            action="store_true",
            default=False,
            help="create a sample for each protein sequence annotated for given EC",
        )
        parser.add_argument(
            "--sample_uniprot_per_ec",
            action="store_true",
            default=False,
            help="randomly sample a uniprot for each EC at getitem",
        )
        parser.add_argument(
            "--remove_stereochemistry",
            action="store_true",
            default=False,
            help="remove stereochemistry from smiles",
        )
        parser.add_argument(
            "--min_reaction_quality",
            type=float,
            default=-1,
            help="minimum threshold to use for filtering reactions based on quality score",
        )
        parser.add_argument(
            "--load_wln_cache_in_dataset",
            action="store_true",
            default=False,
            help="load cache for wln in getitem",
        )
        parser.add_argument(
            "--split_multiproduct_samples",
            action="store_true",
            default=False,
            help="split products into different samples",
        )
        parser.add_argument(
            "--use_one_hot_mol_features",
            action="store_true",
            default=False,
            help="encode node and edge features of molecule as one-hot",
        )
        parser.add_argument(
            "--scaffold_balanced",
            action="store_true",
            default=False,
            help="balance the scaffold sets",
        )
        parser.add_argument(
            "--version",
            type=str,
            default="1",
            help="enzyme map version number",
        )
        parser.add_argument(
            "--convert_graph_to_smiles",
            action="store_true",
            default=False,
            help="for sequence based methods",
        )
        parser.add_argument(
            "--reaction_to_products_dir",
            type=str,
            default=None,
            help="cache for post process step",
        )
        parser.add_argument(
            "--remove_duplicate_reactions",
            action="store_true",
            default=False,
            help="remove duplicates",
        )

    @property
    def SUMMARY_STATEMENT(self) -> None:
        try:
            reactions = [
                "{}>>{}".format(".".join(d["reactants"]), ".".join(d["products"]))
                for d in self.dataset
            ]
        except:
            reactions = "NA"
        try:
            proteins = [d["uniprot_id"] for d in self.dataset]
        except:
            proteins = "NA"
        try:
            ecs = [d["ec"] for d in self.dataset]
        except:
            ecs = "NA"
        statement = f""" 
        * Number of reactions: {len(set(reactions))}
        * Number of proteins: {len(set(proteins))}
        * Number of ECs: {len(set(ecs))}
        """
        return statement


@register_object("enzymemap_substrate", "dataset")
class EnzymeMapSubstrate(EnzymeMap):
    def __init__(self, args, split_group) -> None:
        super(EnzymeMapSubstrate, EnzymeMapSubstrate).__init__(self, args, split_group)

    def create_dataset(
        self, split_group: Literal["train", "dev", "test"]
    ) -> List[dict]:
        # if removing top K
        # if self.args.topk_byproducts_to_remove is not None:
        #     raw_byproducts = Counter([r for d in self.metadata_json for r in d["products"]]).most_common(self.args.topk_byproducts_to_remove)
        #     mapped_byproducts = Counter([r for d in self.metadata_json for r in d.get("mapped_products", []) ]).most_common(self.args.topk_byproducts_to_remove)
        #     self.common_byproducts = {s[0]:True for byproducts in [raw_byproducts, mapped_byproducts] for s in byproducts}
        self.mol2size = {}

        if self.args.topk_substrates_to_remove is not None:
            raw_substrates = Counter(
                [r for d in self.metadata_json for r in d["reactants"]]
            ).most_common(self.args.topk_substrates_to_remove)
            self.common_substrates = set([s[0] for s in raw_substrates])

        dataset = []
        seen_before = set()
        # self.metadata_json = self.metadata_json[:100] + self.metadata_json[-100:] + self.metadata_json[70711:70811]

        for rowid, reaction in tqdm(
            enumerate(self.metadata_json),
            desc="Building dataset",
            total=len(self.metadata_json),
            ncols=100,
        ):
            self.mol2size = {}

            ec = reaction["ec"]
            reactants = (
                sorted(reaction.get("mapped_reactants", []))
                if self.args.use_mapped_reaction
                else sorted(reaction["reactants"])
            )
            products = (
                sorted(reaction.get("mapped_products", []))
                if self.args.use_mapped_reaction
                else sorted(reaction["products"])
            )
            products = [p for p in products if p not in reactants]

            # if self.args.topk_byproducts_to_remove is not None:
            #     products = [p for p in products if p not in self.common_byproducts]
            if self.args.topk_substrates_to_remove is not None:
                reactants = [s for s in reactants if s not in self.common_substrates]

            valid_uniprots = set()
            for r in reactants:
                for uniprot in self.ec2uniprot.get(ec, []):
                    temp_sample = {
                        "smiles": r,
                        "ec": ec,
                        "protein_id": uniprot,
                        "uniprot_id": uniprot,
                        "sequence": self.uniprot2sequence[uniprot],
                        "y": 1,
                    }
                    if self.skip_sample(temp_sample, split_group):
                        continue

                    valid_uniprots.add(uniprot)

                if len(valid_uniprots) == 0:
                    continue

                # whether or not to use all of the sequences
                if self.args.use_all_sequences:
                    for uniprot in valid_uniprots:
                        sample = {
                            "smiles": r,
                            "ec": ec,
                            "rowid": f"{uniprot}_{reaction['rxnid']}",
                            "sample_id": f"{uniprot}_{reaction['rxnid']}",
                            "uniprot_id": uniprot,
                            "protein_id": uniprot,
                            "y": 1,
                        }
                        # remove duplicate prot-substrate pairs
                        if f"{uniprot}_{r}" in seen_before:
                            continue
                        seen_before.add(f"{uniprot}_{r}")
                        # add reaction sample to dataset
                        try:
                            if self.args.use_protein_graphs:
                                graph_path = os.path.join(
                                    self.args.protein_graphs_dir,
                                    "processed",
                                    f"{sample['uniprot_id']}_graph.pt",
                                )

                                if not os.path.exists(graph_path):
                                    data = self.create_protein_graph(sample)
                                    if data is None:
                                        raise Exception(
                                            "Could not generate protein graph"
                                        )
                                    torch.save(data, graph_path)
                                # data created
                                # node embedding dimension is not the same as expected by the model
                                elif (
                                    data["receptor"].x.shape[1]
                                    != ESM_MODEL2HIDDEN_DIM[self.esm_dir.split("/")[-1]]
                                ):
                                    print(
                                        "Node embedding dimension is not the same as expected by the model"
                                    )
                                    data = self.create_protein_graph(sample)
                                    if data is None:
                                        raise Exception(
                                            "Could not generate protein graph"
                                        )
                                    torch.save(data, graph_path)
                            dataset.append(sample)

                        except Exception as e:
                            print(
                                f"Error processing {sample['sample_id']} because of {e}"
                            )
                            continue
                else:
                    assert (
                        not self.args.add_neg_for_all_substrates
                    ), "Cannot use all sequences and add negative samples for all substrates"
                    # TODO: can loop over the following k times if you want k sequences per sample
                    # instead of adding one uniprot, store all uniprots and then sample in getitem
                    for split in ["train", "dev", "test"]:
                        valid_uniprots_split = [
                            u for u in valid_uniprots if self.uniprot2split[u] == split
                        ]
                        if len(valid_uniprots_split) == 0:
                            continue
                        # want to evaluate on full dev and test
                        if split in ["dev", "test"] and self.args.eval_on_full_dev_test:
                            for prot in valid_uniprots_split:
                                if f"{prot}_{r}" in seen_before:
                                    continue
                                seen_before.add(f"{prot}_{r}")
                                sample = {
                                    "smiles": r,
                                    "ec": ec,
                                    "sample_id": f"{hashlib.sha256(r.encode('utf-8')).hexdigest()}_{hashlib.sha256(str(sorted(list([prot]))).encode('utf-8')).hexdigest()}",
                                    "uniprot_ids": list([prot]),
                                    "protein_ids": list([prot]),
                                    "y": 1,
                                    "split": split,
                                }

                                # add reaction sample to dataset
                                try:
                                    if self.args.use_protein_graphs:
                                        for u_prot_id in sample["uniprot_ids"]:
                                            graph_path = os.path.join(
                                                self.args.protein_graphs_dir,
                                                "processed",
                                                f"{u_prot_id}_graph.pt",
                                            )

                                            if not os.path.exists(graph_path):
                                                data = self.create_protein_graph(sample)
                                                if data is None:
                                                    raise Exception(
                                                        "Could not generate protein graph"
                                                    )
                                                torch.save(data, graph_path)
                                    dataset.append(sample)

                                except Exception as e:
                                    print(
                                        f"Error processing {sample['sample_id']} because of {e}"
                                    )
                                    continue
                        else:
                            sample = {
                                "smiles": r,
                                "ec": ec,
                                "sample_id": f"{hashlib.sha256(r.encode('utf-8')).hexdigest()}_{hashlib.sha256(str(sorted(list(valid_uniprots_split))).encode('utf-8')).hexdigest()}",
                                "uniprot_ids": list(valid_uniprots_split),
                                "protein_ids": list(valid_uniprots_split),
                                "y": 1,
                                "split": split,
                            }
                            # remove duplicate prot-substrate pairs
                            skip_uniprots = []
                            for uniprot in valid_uniprots_split:
                                if f"{uniprot}_{r}" in seen_before:
                                    skip_uniprots.append(uniprot)
                                seen_before.add(f"{uniprot}_{r}")
                            sample["uniprot_ids"] = [
                                u
                                for u in sample["uniprot_ids"]
                                if u not in skip_uniprots
                            ]
                            sample["protein_ids"] = [
                                u
                                for u in sample["protein_ids"]
                                if u not in skip_uniprots
                            ]
                            if len(sample["uniprot_ids"]) == 0:
                                continue

                            # add reaction sample to dataset
                            try:
                                if self.args.use_protein_graphs:
                                    for u_prot_id in sample["uniprot_ids"]:
                                        graph_path = os.path.join(
                                            self.args.protein_graphs_dir,
                                            "processed",
                                            f"{u_prot_id}_graph.pt",
                                        )

                                        if not os.path.exists(graph_path):
                                            data = self.create_protein_graph(sample)
                                            if data is None:
                                                raise Exception(
                                                    "Could not generate protein graph"
                                                )
                                            torch.save(data, graph_path)
                                dataset.append(sample)

                            except Exception as e:
                                print(
                                    f"Error processing {sample['sample_id']} because of {e}"
                                )
                                continue

                    # TODO: might be missing sequences from ECs...

        return dataset

    def __getitem__(self, index):
        sample = self.dataset[index]

        try:
            if self.args.use_all_sequences:
                # ec = sample["ec"]
                uniprot_id = sample["uniprot_id"]
                sequence = self.uniprot2sequence[uniprot_id]

                if sample["y"] == 0 and self.args.sample_negatives_on_get:
                    # sample a negative substrate
                    if len(self.prot_id_to_negatives[uniprot_id]) == 0:
                        print("This protein has no negatives")
                        return None
                    sample["smiles"] = list(self.prot_id_to_negatives[uniprot_id])[
                        np.random.randint(0, len(self.prot_id_to_negatives[uniprot_id]))
                    ]

            else:
                uniprot_id = sample["uniprot_ids"][
                    np.random.randint(0, len(sample["uniprot_ids"]))
                ]
                sequence = self.uniprot2sequence[uniprot_id]

            smiles = sample["smiles"]

            if self.args.use_random_smiles_representation:
                try:
                    smiles = randomize_smiles_rotated(smiles)
                except:
                    pass

            sample_id = sample["sample_id"]
            item = {
                "sequence": sequence,
                # "ec": ec,
                # "organism": sample.get("organism", "none"),
                "protein_id": uniprot_id,
                "uniprot_id": uniprot_id,
                "sample_id": sample_id,
                "smiles": smiles,
                "y": sample["y"],
            }
            if (
                "split" in sample
                or self.args.split_type == "mmseqs_precomputed"
                or self.args.split_type == "scaffold"
            ):
                item["split"] = sample["split"]

            if self.args.use_pesto_scores:
                scores = self.get_pesto_scores(item["protein_id"])
                if (scores is None) or (scores.shape[0] != len(item["sequence"])):
                    # make all zeros of length sequence
                    scores = torch.zeros(len(item["sequence"]))
                item["sequence_annotation"] = scores

                if self.args.replace_seq_with_smiles:
                    item["pesto_indices"] = np.where(
                        item["sequence_annotation"] >= self.args.pesto_threshold
                    )[0]
                    if len(item["pesto_indices"]) == 0:
                        # if all pesto scores are low, just use the entire protein
                        item["pesto_indices"] = np.where(
                            item["sequence_annotation"] >= 0
                        )[0]
                    item["pesto_sequence"] = "".join(
                        [item["sequence"][i] for i in item["pesto_indices"]]
                    )
                    item["sequence_smiles"] = []
                    for letter in item["pesto_sequence"]:
                        item["sequence_smiles"].append(AA_TO_SMILES.get(letter, None))

                    item["sequence_smiles"] = [
                        from_smiles(s) for s in item["sequence_smiles"] if s is not None
                    ]
                    item["sequence_smiles"] = Batch.from_data_list(
                        item["sequence_smiles"]
                    )

            if self.args.use_protein_graphs:
                # load the protein graph
                graph_path = os.path.join(
                    self.args.protein_graphs_dir,
                    "processed",
                    f"{item['uniprot_id']}_graph.pt",
                )
                data = torch.load(graph_path)
                if data is None:
                    structure_path = os.path.join(
                        self.args.protein_structures_dir,
                        f"AF-{item['uniprot_id']}-F1-model_v4.cif",
                    )
                    assert os.path.exists(
                        structure_path
                    ), f"Structure path {graph_path} does not exist"
                    print(
                        f"Structure path does exist, but graph path does not exist {graph_path}"
                    )
                    data = self.create_protein_graph(item)
                    torch.save(data, graph_path)

                data = self.add_additional_data_to_graph(data, item)
                if hasattr(data, "x") and not hasattr(data["receptor"], "x"):
                    data["receptor"].x = data.x

                keep_keys = {
                    "receptor",
                    "mol_data",
                    "sequence",
                    "protein_id",
                    "uniprot_id",
                    "sample_id",
                    "smiles",
                    "y",
                    ("receptor", "contact", "receptor"),
                }

                data_keys = data.to_dict().keys()
                for d_key in data_keys:
                    if not d_key in keep_keys:
                        delattr(data, d_key)

                coors = data["receptor"].pos
                feats = data["receptor"].x
                edge_index = data["receptor", "contact", "receptor"].edge_index
                assert (
                    coors.shape[0] == feats.shape[0]
                ), f"Number of nodes do not match between coors ({coors.shape[0]}) and feats ({feats.shape[0]})"

                assert (
                    max(edge_index[0]) < coors.shape[0]
                    and max(edge_index[1]) < coors.shape[0]
                ), "Edge index contains node indices not present in coors"

                return data
            else:  # just the substrate, with the protein sequence in the Data object
                reactant = from_smiles(sample["smiles"])
                for key in item.keys():
                    reactant[key] = item[key]
                return reactant

        except Exception as e:
            print(f"Getitem: Could not load sample: {sample['sample_id']} due to {e}")

    def add_additional_data_to_graph(self, data, sample):
        skipped_keys = set(["protein_path", "embedding_path"])
        for key in sample.keys():
            if not key in skipped_keys and key not in data.to_dict().keys():
                data[key] = sample[key]
        data["mol_data"] = from_smiles(sample["smiles"])
        return data

    def post_process(self, args):
        if args.sample_negatives:
            self.dataset = self.add_negatives(
                self.dataset, split_group=self.split_group
            )

    def add_negatives(self, dataset, split_group):
        # # Uncomment to add ec
        # uniprot2ec = {}
        # for s in dataset:
        #     uniprot2ec[s["uniprot_id"]] = s["ec"]
        all_substrates = set(d["smiles"] for d in dataset)
        all_substrates_list = list(all_substrates)
        if self.args.use_all_sequences:
            all_uniprots = set(d["protein_id"] for d in dataset)
        else:
            all_uniprots = set(uniprot for d in dataset for uniprot in d["uniprot_ids"])

        # filter out negatives based on some metric (e.g. similarity)
        if self.args.sample_negatives_range is not None:
            min_sim, max_sim = self.args.sample_negatives_range

            # get features of every smile
            smiles2feature = {
                smile: get_rdkit_feature(mol=smile, method="morgan_binary")
                for smile in all_substrates
            }
            # normalize and stack all feature vecs
            smile_fps = np.array(
                [
                    smiles2feature[smile] / np.linalg.norm(smiles2feature[smile])
                    for smile in all_substrates
                ]
            )
            # similarity matrix
            smile_similarity = smile_fps @ smile_fps.T

            # this is a dict of each molecule and the similar but different molecules in the range defined
            smiles2negatives = defaultdict(set)
            for smi_i, (smile, sim_row) in tqdm(
                enumerate(zip(all_substrates_list, smile_similarity)),
                desc="Retrieving all negatives",
                total=len(all_substrates_list),
            ):
                # find where in the row of molecules which indices are above similarity threshold
                valid_indices = np.where((sim_row > min_sim) & (sim_row < max_sim))[0]
                # add to the dict
                smiles2negatives[smile].update(
                    all_substrates_list[j] for j in valid_indices
                )

        # this dict holds each uniprot and the molecules that it is not a binder for in the similarity threshold
        # note that it the second loop must run to remove positives
        self.prot_id_to_negatives = defaultdict(set)
        for sample in tqdm(dataset, desc="Sampling negatives"):
            if self.args.use_all_sequences:
                if self.args.sample_negatives_range is not None:
                    prot_id = sample["protein_id"]
                    self.prot_id_to_negatives[prot_id].update(
                        smiles2negatives[sample["smiles"]]
                    )
                else:
                    self.prot_id_to_negatives.update(all_substrates_list)
            else:
                for uniprot in sample["uniprot_ids"]:
                    if self.args.sample_negatives_range is not None:
                        self.prot_id_to_negatives[uniprot].update(
                            smiles2negatives[sample["smiles"]]
                        )
                    else:
                        self.prot_id_to_negatives[uniprot].update(all_substrates_list)

        for sample in tqdm(dataset):
            if self.args.use_all_sequences:
                prot_id = sample["protein_id"]
                # remove the current smile from the dict
                self.prot_id_to_negatives[prot_id].discard(sample["smiles"])
            else:
                for uniprot in sample["uniprot_ids"]:
                    self.prot_id_to_negatives[uniprot].discard(sample["smiles"])

        smile2negative_prot = defaultdict(set)
        for prot, negatives in tqdm(
            self.prot_id_to_negatives.items(),
            desc="Computing negatives for all substrates",
        ):
            for smile in negatives:
                smile2negative_prot[smile].add(prot)

        rowid = len(dataset)
        prot_id2positive_smiles = defaultdict(
            set
        )  # this is for later to add missing negatives
        prots_with_no_negatives = []
        negatives_to_add = []
        no_negatives = 0
        # now that we have all of the negatives found, add them to the dataset
        for sample in tqdm(dataset, desc="Processing negatives"):
            if self.args.use_all_sequences:
                negatives = self.prot_id_to_negatives[sample["protein_id"]]
                prot_id = sample["protein_id"]
                prot_id2positive_smiles[prot_id].add(sample["smiles"])
                if len(negatives) == 0:
                    no_negatives += 1
                    prots_with_no_negatives.append(prot_id)
                    continue

                # sometimes we dont want to add all negative options, just pick k
                if self.args.sample_k_negatives is not None:
                    if len(negatives) < self.args.sample_k_negatives:
                        new_negatives = list(negatives)
                    else:
                        new_negatives = random.sample(
                            list(negatives), self.args.sample_k_negatives
                        )
                else:
                    new_negatives = list(negatives)

                for rid, reactant in enumerate(new_negatives):
                    item = {
                        # "ec": ec,
                        "protein_id": prot_id,
                        "uniprot_id": prot_id,
                        "sample_id": prot_id + "_" + str(rowid + rid),
                        "smiles": reactant,
                        # "split": uniprot2split[prot_id],
                        "y": 0,
                    }
                    if self.args.split_type == "scaffold":
                        item["split"] = self.to_split[item["smiles"]]

                    if self.skip_sample(item, split_group):
                        continue

                    negatives_to_add.append(item)

                rowid += len(new_negatives)
            else:
                negative_uniprots = smile2negative_prot[sample["smiles"]]
                for split in ["train", "dev", "test"]:
                    if split in ["dev", "test"] and self.args.eval_on_full_dev_test:
                        valid_uniprots_split = [
                            u
                            for u in negative_uniprots
                            if self.uniprot2split[u] == split
                        ]
                        if len(valid_uniprots_split) == 0:
                            continue
                        for prot in valid_uniprots_split:
                            item = {
                                # "ec": ec,
                                "protein_ids": [prot],
                                "uniprot_ids": [prot],
                                "sample_id": f"{hashlib.sha256(smile.encode('utf-8')).hexdigest()}_{hashlib.sha256(str(sorted(list([prot]))).encode('utf-8')).hexdigest()}",
                                "smiles": smile,
                                "split": split,
                                "y": 0,
                            }
                            if self.skip_sample(item, split_group):
                                continue

                            negatives_to_add.append(item)
                    else:
                        valid_uniprots_split = [
                            u
                            for u in negative_uniprots
                            if self.uniprot2split[u] == split
                        ]
                        if len(valid_uniprots_split) == 0:
                            continue
                        item = {
                            # "ec": ec,
                            "protein_ids": valid_uniprots_split,
                            "uniprot_ids": valid_uniprots_split,
                            "sample_id": f"{hashlib.sha256(smile.encode('utf-8')).hexdigest()}_{hashlib.sha256(str(sorted(list(valid_uniprots_split))).encode('utf-8')).hexdigest()}",
                            "smiles": smile,
                            "split": split,
                            "y": 0,
                        }
                        if self.skip_sample(item, split_group):
                            continue

                        negatives_to_add.append(item)

        if self.args.add_neg_for_all_substrates:  # and protein
            missing_substrates = []
            for mol in all_substrates:
                if not mol in smile2negative_prot:
                    missing_substrates.append(mol)
                else:
                    prot = random.sample(smile2negative_prot[mol], 1)[0]
                    sample = {
                        "protein_id": prot,
                        "uniprot_id": prot,
                        "sample_id": prot + "_" + str(rowid + rid),
                        "smiles": mol,
                        "y": 0,
                    }
                    if self.args.split_type == "scaffold":
                        sample["split"] = self.to_split[sample["smiles"]]

                    rid += 1
                    negatives_to_add.append(sample)

            for uniprot in prots_with_no_negatives:
                mol = random.sample(all_substrates, 1)[0]
                i = 0
                # if you happened to pick a positive pair, try to get a negative one 20 times
                while mol in prot_id2positive_smiles[uniprot] and i < 20:
                    i += 1
                    mol = random.sample(all_substrates, 1)[0]
                if i == 20:
                    continue  # just skip this one
                sample = {
                    "protein_id": prot,
                    "uniprot_id": prot,
                    "sample_id": prot + "_" + str(rowid + rid),
                    "smiles": mol,
                    "y": 0,
                }
                if self.args.split_type == "scaffold":
                    sample["split"] = self.to_split[sample["smiles"]]
                rid += 1
                negatives_to_add.append(sample)

            for substr in missing_substrates:
                prot = random.sample(all_uniprots, 1)[0]
                i = 0
                # if you happened to pick a positive pair, try to get a negative one 20 times
                while substr in prot_id2positive_smiles[prot] and i < 20:
                    i += 1
                    prot = random.sample(all_uniprots, 1)[0]
                if i == 20:
                    continue  # just skip this one
                sample = {
                    "protein_id": prot,
                    "uniprot_id": prot,
                    "sample_id": prot + "_" + str(rowid + rid),
                    "smiles": substr,
                    "y": 0,
                }
                if self.args.split_type == "scaffold":
                    sample["split"] = self.to_split[sample["smiles"]]
                rid += 1
                negatives_to_add.append(sample)

        print(f"[magenta] Adding {len(negatives_to_add)} negatives [/magenta]")
        print(f"[magenta] Missing any negatives for {no_negatives} ECs [/magenta]")
        print(f"[magenta] Total number of positive samples: {len(dataset)} [/magenta]")
        dataset += negatives_to_add

        return dataset

    def skip_sample(self, sample, split_group) -> bool:
        if "ec" in sample and "-" in sample["ec"]:
            return True

        if self.args.use_protein_graphs:
            structures_dir = os.path.join(
                self.args.protein_structures_dir,
                f"AF-{sample['uniprot_id']}-F1-model_v4.cif",
            )

            if not os.path.exists(structures_dir):
                return True

        # if sequence is unknown
        if "sequence" in sample:
            if (sample["sequence"] is None) or (len(sample["sequence"]) == 0):
                return True

            if (self.args.max_protein_length is not None) and len(
                sample["sequence"]
            ) > self.args.max_protein_length:
                return True

        mol = sample["smiles"]
        if not (mol in self.mol2size):
            self.mol2size[mol] = rdkit.Chem.MolFromSmiles(mol).GetNumAtoms()

        if self.args.max_reactant_size is not None:
            if self.mol2size[mol] > self.args.max_reactant_size:
                return True

        if self.mol2size[mol] < 2:
            return True

        return False

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args"""
        super(EnzymeMapSubstrate, EnzymeMapSubstrate).add_args(parser)
        parser.add_argument(
            "--replace_seq_with_smiles",
            action="store_true",
            default=False,
            help="whether to replace sequence with smiles",
        )
        parser.add_argument(
            "--pesto_threshold",
            type=float,
            default=0.5,
            help="threshold for pesto predictions",
        )
        parser.add_argument(
            "--sample_negatives",
            action="store_true",
            default=False,
            help="whether to sample negative substrates",
        )
        parser.add_argument(
            "--topk_substrates_to_remove",
            type=int,
            default=None,
            help="remove common substrates",
        )
        parser.add_argument(
            "--sample_negatives_range",
            type=float,
            nargs=2,
            default=None,
            help="range of similarity to sample negatives from",
        )
        parser.add_argument(
            "--precomputed_esm_features_dir",
            type=str,
            default=None,
            help="directory with precomputed esm features for computation efficiency",
        )
        parser.add_argument(
            "--sample_k_negatives",
            type=int,
            default=None,
            help="number of negatives to sample from each ec",
        )
        parser.add_argument(
            "--sample_negatives_on_get",
            action="store_true",
            default=False,
            help="whether to sample negatives on get",
        )
        parser.add_argument(
            "--add_neg_for_all_substrates",
            action="store_true",
            default=False,
            help="whether to add negatives for all substrates",
        )
        parser.add_argument(
            "--use_all_sequences",
            action="store_true",
            default=False,
            help="whether to add a sample for every sequence or one sequence per substrate",
        )
        parser.add_argument(
            "--eval_on_full_dev_test",
            action="store_true",
            default=False,
            help="whether to evaluate on full (all proteins) dev and test sets",
        )


@register_object("enzymemap_reaction_graph", "dataset")
class EnzymeMapGraph(EnzymeMap):
    def post_process(self, args):
        def make_reaction_to_products():
            reaction_to_products = defaultdict(set)
            if args.create_sample_per_sequence:
                key = lambda sample: f"{sample['ec']}{'.'.join(sample['reactants'])}"
            else:
                key = lambda sample: ".".join(sample["reactants"])
            for sample in tqdm(self.dataset, desc="post-processing", ncols=100):
                reaction_to_products[key(sample)].add(
                    (
                        ".".join(sample["products"]),
                        stringify_sets(sorted(sample["bond_changes"])),
                    )
                )
            return reaction_to_products

        # add all possible products
        # if args.reaction_to_products_dir is not None:
        #     if not os.path.exists(args.reaction_to_products_dir):
        #         os.makedirs(args.reaction_to_products_dir)
        #     path = f"{args.reaction_to_products_dir}/{self.split_group}.p"
        #     if os.path.exists(path):
        #         self.reaction_to_products = pickle.load(open(path, "rb"))
        #     else:
        #         self.reaction_to_products = make_reaction_to_products()
        #         pickle.dump(self.reaction_to_products, open(path, "wb"))
        # else:
        #     self.reaction_to_products = make_reaction_to_products()

        # set ec levels to id for use in modeling
        ecs = set(d["ec"] for d in self.dataset)
        ecs = [e.split(".") for e in ecs]
        args.ec_levels = {}
        for level in range(1, 5, 1):
            unique_classes = sorted(list(set(".".join(ec[:level]) for ec in ecs)))
            args.ec_levels[str(level)] = {c: i for i, c in enumerate(unique_classes)}

    def create_dataset(
        self, split_group: Literal["train", "dev", "test"]
    ) -> List[dict]:
        # if removing top K
        if self.args.topk_byproducts_to_remove is not None:
            raw_byproducts = Counter(
                [r for d in self.metadata_json for r in d["products"]]
            ).most_common(self.args.topk_byproducts_to_remove)
            mapped_byproducts = Counter(
                [r for d in self.metadata_json for r in d.get("mapped_products", [])]
            ).most_common(self.args.topk_byproducts_to_remove)
            self.common_byproducts = {
                s[0]: True
                for byproducts in [raw_byproducts, mapped_byproducts]
                for s in byproducts
            }

        if self.args.remove_duplicate_reactions:
            self.unique_reactions = set()

        dataset = []

        rkey = (
            "mapped_reactants"
            if "mapped_reactants" in self.metadata_json[0]
            else "reactants"
        )
        pkey = (
            "mapped_products"
            if "mapped_products" in self.metadata_json[0]
            else "products"
        )

        self.mol2size = {}

        for rowid, reaction in tqdm(
            enumerate(self.metadata_json),
            desc="Building dataset",
            total=len(self.metadata_json),
            ncols=100,
        ):
            ec = reaction["ec"]
            eclevels_dict = {
                f"ec{ec_level+1}": ".".join(ec.split(".")[: (ec_level + 1)])
                for ec_level, _ in enumerate(ec.split("."))
            }
            organism = reaction.get("organism", "")

            reactants = sorted([s for s in reaction[rkey] if s != "[H+]"])
            products = sorted([s for s in reaction[pkey] if s != "[H+]"])
            products = [p for p in products if p not in reactants]

            if self.args.topk_byproducts_to_remove is not None:
                products = [p for p in products if p not in self.common_byproducts]

            reaction_string = "{}>>{}".format(".".join(reactants), ".".join(products))

            bond_changes = reaction.get("bond_changes", None)
            if not bond_changes:
                try:
                    bond_changes = get_bond_changes(reaction_string)
                except:
                    continue

            # select uniprots
            if self.args.version == "1":
                alluniprots = self.ec2uniprot.get(ec, [])
                protein_refs = []
            elif self.args.version == "2":
                protein_refs = eval(reaction["protein_refs"])
                alluniprots = protein_refs
                if (len(alluniprots) == 0) and self.args.sample_uniprot_per_ec:
                    alluniprots = self.ec2uniprot.get(ec, [])

            if self.args.create_sample_per_sequence or self.args.sample_uniprot_per_ec:
                for uniprot in alluniprots:
                    sample = {
                        "reaction_string": "{}>>{}".format(
                            ".".join(sorted(reaction["reactants"])),
                            ".".join(sorted(reaction["products"])),
                        ),
                        "df_row": rowid,
                        "quality": reaction["quality"],
                        "reactants": reactants,
                        "products": products,
                        "ec": ec,
                        "rowid": reaction["rxnid"],
                        "sample_id": f"{uniprot}_{reaction['rxnid']}_{rowid}",
                        "uniprot_id": uniprot,
                        "protein_id": uniprot,
                        "bond_changes": list(bond_changes),
                        "split": reaction.get("split", None),
                        "protein_refs": protein_refs,
                        "protein_db": reaction.get("protein_db", ""),
                        "rule_id": reaction["rule_id"],
                    }
                    sample.update(eclevels_dict)

                    if self.skip_sample(sample, split_group):
                        continue

                    if self.args.sample_uniprot_per_ec:
                        self.valid_ec2uniprot[ec].add(uniprot)

                        sample = {
                            "df_row": rowid,
                            "quality": reaction["quality"],
                            "reactants": reactants,
                            "products": products,
                            "ec": ec,
                            "rowid": reaction["rxnid"],
                            "sample_id": str(reaction["rxnid"]),
                            "uniprot_id": "",
                            "protein_id": "",
                            "sequence": "X",
                            "bond_changes": list(bond_changes),
                            "split": reaction.get("split", None),
                            "protein_refs": protein_refs,
                            "rule_id": reaction["rule_id"],
                        }
                        sample.update(eclevels_dict)

                        if self.args.split_type == "ec_hold_out":
                            unique_sample_content = f"{reaction_string}"
                            hashed_sample_content = hashlib.sha256(
                                unique_sample_content.encode("utf-8")
                            ).hexdigest()
                            sample["hash_sample_id"] = hashed_sample_content

                        if self.args.split_multiproduct_samples:
                            for product_id, p in enumerate(products):
                                psample = copy.deepcopy(sample)
                                psample["products"] = [p]
                                psample["sample_id"] += f"_{product_id}"
                                dataset.append(psample)

                        else:
                            dataset.append(sample)

                    else:
                        if self.args.split_type == "ec_hold_out":
                            unique_sample_content = (
                                f"{reaction_string}{uniprot}{organism}"
                            )
                            hashed_sample_content = hashlib.sha256(
                                unique_sample_content.encode("utf-8")
                            ).hexdigest()
                            sample["hash_sample_id"] = hashed_sample_content

                        try:
                            # make prot graph if missing
                            if self.args.use_protein_graphs:
                                graph_path = os.path.join(
                                    self.args.protein_graphs_dir,
                                    "processed",
                                    f"{sample['uniprot_id']}_graph.pt",
                                )
                                structure_path = os.path.join(
                                    self.args.protein_structures_dir,
                                    f"AF-{sample['uniprot_id']}-F1-model_v4.cif",
                                )
                                if not os.path.exists(structure_path):
                                    continue
                                if not os.path.exists(graph_path):
                                    print("Generating none existent protein graph")
                                    data = self.create_protein_graph(sample)
                                    if data is None:
                                        raise Exception(
                                            "Could not generate protein graph"
                                        )
                                    torch.save(data, graph_path)

                        except Exception as e:
                            print(
                                f"Error processing {sample['sample_id']} because of {e}"
                            )
                            continue

                        # add reaction sample to dataset
                        if self.args.split_multiproduct_samples:
                            for product_id, p in enumerate(products):
                                psample = copy.deepcopy(sample)
                                psample["products"] = [p]
                                psample["sample_id"] += f"_{product_id}"
                                preaction_string = "{}>>{}".format(
                                    ".".join(psample["reactants"]), p
                                )
                                uniprot = psample["uniprot_id"]
                                punique_sample_content = (
                                    f"{preaction_string}{uniprot}{psample['organism']}"
                                )
                                phashed_sample_content = hashlib.sha256(
                                    punique_sample_content.encode("utf-8")
                                ).hexdigest()
                                psample["hash_sample_id"] = phashed_sample_content
                                dataset.append(psample)
                        else:
                            dataset.append(sample)
            else:
                sample = {
                    "df_row": rowid,
                    "quality": reaction["quality"],
                    "reactants": reactants,
                    "products": products,
                    "ec": ec,
                    "rowid": reaction["rxnid"],
                    "sample_id": str(reaction["rxnid"]),
                    "uniprot_id": "",
                    "protein_id": "",
                    "sequence": "X",
                    "bond_changes": list(bond_changes),
                    "split": reaction.get("split", None),
                    "protein_refs": protein_refs,
                    "rule_id": reaction["rule_id"],
                }

                unique_sample_content = f"{reaction_string}"
                hashed_sample_content = hashlib.sha256(
                    unique_sample_content.encode("utf-8")
                ).hexdigest()
                sample["hash_sample_id"] = hashed_sample_content

                sample.update(eclevels_dict)

                if self.skip_sample(sample, split_group):
                    continue

                if self.args.split_multiproduct_samples:
                    for product_id, p in enumerate(products):
                        psample = copy.deepcopy(sample)
                        psample["products"] = [p]
                        psample["sample_id"] += f"_{product_id}"
                        dataset.append(psample)

                else:
                    dataset.append(sample)

        return dataset

    def __getitem__(self, index):
        sample = self.dataset[index]

        try:
            reactants, products = copy.deepcopy(sample["reactants"]), copy.deepcopy(
                sample["products"]
            )

            ec = sample["ec"]
            if self.args.create_sample_per_sequence:
                uniprot_id = sample["uniprot_id"]
                sequence = self.uniprot2sequence.get(uniprot_id, "<unk>")
            elif self.args.sample_uniprot_per_ec:
                valid_uniprots = self.valid_ec2uniprot.get(ec, ["<unk>"])
                uniprot_id = random.sample(valid_uniprots, 1)[0]
                sequence = self.uniprot2sequence[uniprot_id]
            else:
                uniprot_id = "unk"
                sequence = "<unk>"

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

            reactants, atom_map2new_index = from_mapped_smiles(
                ".".join(reactants),
                encode_no_edge=True,
                use_one_hot_encoding=self.args.use_one_hot_mol_features,
            )
            products, _ = from_mapped_smiles(
                ".".join(products),
                encode_no_edge=True,
                use_one_hot_encoding=self.args.use_one_hot_mol_features,
            )

            bond_changes = [
                (atom_map2new_index[int(u)], atom_map2new_index[int(v)], btype)
                for u, v, btype in sample["bond_changes"]
            ]
            bond_changes = [(min(x, y), max(x, y), t) for x, y, t in bond_changes]
            reactants.bond_changes = bond_changes
            sample_id = sample["sample_id"]
            rowid = sample["rowid"]

            # convert bond changes for all_smiles
            # all_smiles_key = (
            #     f"{ec}{reactants.smiles}"
            #     if self.args.create_sample_per_sequence
            #     else reactants.smiles
            # )
            # all_smiles_smiles = [
            #     smiles for smiles, _ in self.reaction_to_products[all_smiles_key]
            # ]
            # all_smiles_bond_changes = [
            #     destringify_sets(bc)
            #     for _, bc in self.reaction_to_products[all_smiles_key]
            # ]
            # all_smiles_bond_changes = [
            #     [
            #         (atom_map2new_index[int(u)], atom_map2new_index[int(v)], btype)
            #         for u, v, btype in changes
            #     ]
            #     for changes in all_smiles_bond_changes
            # ]
            # all_smiles_bond_changes = [
            #     set((min(x, y), max(x, y), t) for x, y, t in bc)
            #     for bc in all_smiles_bond_changes
            # ]
            # all_smiles = [
            #     (prod_smile, prod_bc)
            #     for prod_smile, prod_bc in zip(
            #         all_smiles_smiles, all_smiles_bond_changes
            #     )
            # ]

            reaction_nodes = torch.zeros(reactants.x.shape[0])
            for s in [bond_changes]:
                for u, v, t in s:
                    reaction_nodes[u] = 1
                    reaction_nodes[v] = 1

            reactants.reaction_nodes = reaction_nodes

            item = {
                "x": reaction,
                "reaction": reaction,
                "reactants": reactants,
                "mol": reactants,
                "products": products,
                "sequence": sequence,
                "ec": ec,
                "organism": sample.get("organism", "none"),
                "protein_id": uniprot_id,
                "uniprot_id": uniprot_id,
                "sample_id": sample_id,
                "row_id": rowid,
                "smiles": products,
                "all_smiles": [],  # all_smiles,
                "quality": sample["quality"]
                # "bond_changes": stringify_sets(bond_changes)
            }

            # ecs as tensors
            split_ec = ec.split(".")
            for k, v in self.args.ec_levels.items():
                item[f"ec{k}"] = v.get(".".join(split_ec[: int(k)]), -1)

            if self.args.load_wln_cache_in_dataset:
                item["product_candidates"] = self.cache.get(rowid)

            if self.args.use_pesto_scores:
                scores = self.get_pesto_scores(item["protein_id"])
                if (scores is None) or (scores.shape[0] != len(item["sequence"])):
                    # make all zeros of length sequence
                    scores = torch.zeros(len(item["sequence"]))
                item["sequence_annotation"] = scores

            if self.args.use_protein_graphs:
                # load the protein graph
                graph_path = os.path.join(
                    self.args.protein_graphs_dir,
                    "processed",
                    f"{item['uniprot_id']}_graph.pt",
                )
                data = torch.load(graph_path)
                if data is None:
                    structure_path = os.path.join(
                        self.args.protein_structures_dir,
                        f"AF-{item['uniprot_id']}-F1-model_v4.cif",
                    )
                    assert os.path.exists(
                        structure_path
                    ), f"Structure path {graph_path} does not exist"
                    print(
                        f"Structure path does exist, but graph path does not exist {graph_path} so making graph"
                    )

                    data = self.create_protein_graph(item)
                    torch.save(data, graph_path)

                if hasattr(data, "x") and not hasattr(data["receptor"], "x"):
                    data["receptor"].x = data.x

                keep_keys = {
                    "receptor",
                    # "mol_data",
                    # "sequence",
                    # "protein_id",
                    # "uniprot_id",
                    # "sample_id",
                    # "smiles",
                    # "y",
                    ("receptor", "contact", "receptor"),
                }

                data_keys = data.to_dict().keys()
                for d_key in data_keys:
                    if not d_key in keep_keys:
                        delattr(data, d_key)

                coors = data["receptor"].pos
                feats = data["receptor"].x
                edge_index = data["receptor", "contact", "receptor"].edge_index
                assert (
                    coors.shape[0] == feats.shape[0]
                ), f"Number of nodes do not match between coors ({coors.shape[0]}) and feats ({feats.shape[0]})"

                assert (
                    max(edge_index[0]) < coors.shape[0]
                    and max(edge_index[1]) < coors.shape[0]
                ), "Edge index contains node indices not present in coors"

                item["graph"] = data

            return item

        except Exception as e:
            print(
                f"Could not load sample {sample['uniprot_id']} because of an exception {e}"
            )


@register_object("reaction_graph_inference", "dataset")
class ReactionGraphInference(AbstractDataset):
    def create_dataset(self, split_group: Literal["test"]) -> List[dict]:
        dataset = []

        for rowid, reaction in tqdm(
            enumerate(self.metadata_json),
            desc="Building dataset",
            total=len(self.metadata_json),
            ncols=100,
        ):
            self.mol2size = {}

            ec = reaction["ec"]
            reactants = sorted([s for s in reaction["reactants"] if s != "[H+]"])
            uniprot = reaction["uniprot_id"]
            sequence = reaction["sequence"]

            sample = {
                "reactants": reactants,
                "ec": ec,
                "protein_id": uniprot,
                "sequence": sequence,
            }
            if self.skip_sample(sample, split_group):
                continue

            dataset.append(sample)

        return dataset

    def skip_sample(self, sample, split_group) -> bool:
        # if sequence is unknown
        sequence = sample["sequence"]
        if (sequence is None) or (len(sequence) == 0):
            return True

        if (self.args.max_protein_length is not None) and len(
            sequence
        ) > self.args.max_protein_length:
            return True

        for mol in sample["reactants"]:
            if not (mol in self.mol2size):
                self.mol2size[mol] = rdkit.Chem.MolFromSmiles(mol).GetNumAtoms()

            if self.args.max_reactant_size is not None:
                if self.mol2size[mol] > self.args.max_reactant_size:
                    return True

        return False

    def __getitem__(self, index):
        sample = self.dataset[index]

        try:
            reactants = copy.deepcopy(sample["reactants"])
            ec = sample["ec"]
            uniprot_id = sample["uniprot_id"]
            sequence = self.uniprot2sequence[uniprot_id]
            uniprot_id = sample["uniprot_id"]
            sequence = sample["sequence"]
            reaction = ".".join(reactants)
            reactants = assign_dummy_atom_maps(reaction)
            reactants, atom_map2new_index = from_mapped_smiles(
                ".".join(reactants), encode_no_edge=True
            )
            reactants.bond_changes = []
            sample_id = sample["sample_id"]

            item = {
                "x": reaction,
                "reaction": reaction,
                "reactants": reactants,
                "sequence": sequence,
                "ec": ec,
                "protein_id": uniprot_id,
                "sample_id": sample_id,
                "row_id": sample_id,
            }

            return item

        except Exception as e:
            print(
                f"Could not load sample {sample['uniprot_id']} because of an exception {e}"
            )

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args
        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        super(EnzymeMap, EnzymeMap).add_args(parser)
        parser.add_argument(
            "--max_protein_length",
            type=int,
            default=None,
            help="skip proteins longer than max_protein_length",
        )
        parser.add_argument(
            "--max_reactant_size",
            type=int,
            default=None,
            help="maximum reactant size",
        )
        parser.add_argument(
            "--remove_stereochemistry",
            action="store_true",
            default=False,
            help="remove stereochemistry from smiles",
        )

    @property
    def SUMMARY_STATEMENT(self) -> None:
        try:
            reactions = [".".join(d["reactants"]) for d in self.dataset]
        except:
            reactions = []
        try:
            proteins = [d["uniprot_id"] for d in self.dataset]
        except:
            proteins = []
        try:
            ecs = [d["ec"] for d in self.dataset]
        except:
            ecs = []
        statement = f""" 
        * Number of reactions: {len(set(reactions))}
        * Number of proteins: {len(set(proteins))}
        * Number of ECs: {len(set(ecs))}
        """
        return statement


@register_object("enzymemap_ec", "dataset")
class EnzymeEC(EnzymeMap):
    def create_dataset(self, split_group: Literal["test"]) -> List[dict]:
        uni2ec = defaultdict(set)
        for ec, uniprots in self.ec2uniprot.items():
            for u in uniprots:
                uni2ec[u].add(ec)

        ecs = [d.split(".") for d in self.ec2uniprot]
        unique_classes = sorted(
            list(set(".".join(ec[: self.args.ec_level + 1]) for ec in ecs))
        )
        ec2classid = {c: i for i, c in enumerate(unique_classes)}
        self.args.num_classes = len(ec2classid)

        dataset = []
        for uni, ecs in uni2ec.items():
            y = torch.zeros(len(ec2classid))
            for ec in ecs:
                y[ec2classid[".".join(ec.split(".")[: self.args.ec_level + 1])]] = 1

            sample = {
                "sample_id": uni,
                "protein_id": uni,
                "x": self.uniprot2sequence[uni],
                "y": y,
                "quality": 1,
                "ec": "",
                "sequence": self.uniprot2sequence[uni],
                "reactants": [],
                "products": [],
            }
            if self.skip_sample(sample, split_group):
                continue

            dataset.append(sample)

        return dataset

    def __getitem__(self, index):
        return self.dataset[index]

    def post_process(self, args):
        pass

    @property
    def SUMMARY_STATEMENT(self) -> None:
        labels_per_class = Counter([d["y"].sum().item() for d in self.dataset])
        labels_per_class = {k: labels_per_class[k] for k in sorted(labels_per_class)}
        statement = f""" 
        * Number of classes: {self.args.num_classes}
        * Number of proteins: {len(self.dataset)}
        * Number of labels per class: {labels_per_class}
        """
        return statement


@register_object("enzymemap_drugbank_proteins", "dataset")
class DrugBankProteins(EnzymeMapGraph):
    def create_dataset(self, split_group: Literal["test"]) -> List[dict]:
        dataset = super().create_dataset(split_group)
        drugbank = json.load(
            open(
                "/Mounts/rbg-storage1/users/pgmikhael/DrugBank/drugbank_reactions_with_reactants_itamarupdate.json",
                "r",
            )
        )
        drugbank_proteins = set(u for d in drugbank for u in d["uniprot_ids"])
        dataset = [d for d in dataset if d["protein_id"] in drugbank_proteins]
        return dataset


@register_object("enzymemap_substrate_blip", "dataset")
class EnzymeMapSubstrateBLIP(EnzymeMapSubstrate):
    def create_dataset(
        self, split_group: Literal["train", "dev", "test"]
    ) -> List[dict]:
        self.mol2size = {}
        self.ec_substrate2reaction_center = {}

        if self.args.topk_substrates_to_remove is not None:
            raw_substrates = Counter(
                [r for d in self.metadata_json for r in d["reactants"]]
            ).most_common(self.args.topk_substrates_to_remove)
            self.common_substrates = set([s[0] for s in raw_substrates])

        dataset = []
        seen_before = set()

        for rowid, reaction in tqdm(
            enumerate(self.metadata_json),
            desc="Building dataset",
            total=len(self.metadata_json),
            ncols=100,
        ):
            ec = reaction["ec"]
            reactants = (
                sorted(reaction.get("mapped_reactants", []))
                if self.args.use_mapped_reaction
                else sorted(reaction["reactants"])
            )

            if self.args.topk_substrates_to_remove is not None:
                reactants = [s for s in reactants if s not in self.common_substrates]

            bond_changes = reaction.get("bond_changes", None)
            reaction_nodes = set(
                [int(k[0]) for k in bond_changes] + [int(k[1]) for k in bond_changes]
            )

            # select uniprots
            if self.args.version == "1":
                alluniprots = self.ec2uniprot.get(ec, [])
                protein_refs = []
            elif self.args.version == "2":
                protein_refs = eval(reaction["protein_refs"])
                alluniprots = protein_refs
                if (len(alluniprots) == 0) and self.args.sample_uniprot_per_ec:
                    alluniprots = self.ec2uniprot.get(ec, [])

            valid_uniprots = set()
            for reactant_id, reactant in enumerate(reactants):
                for uniprot in alluniprots:
                    temp_sample = {
                        "smiles": reactant,
                        "ec": ec,
                        "protein_id": uniprot,
                        "uniprot_id": uniprot,
                        "sequence": self.uniprot2sequence.get(uniprot, None),
                        "y": 1,
                        "protein_db": reaction.get("protein_db", ""),
                        "protein_refs": protein_refs,
                        "organism": reaction.get("organism", ""),
                    }
                    if self.skip_sample(temp_sample, split_group):
                        continue

                    valid_uniprots.add(uniprot)

                if len(valid_uniprots) == 0:
                    continue

                # update reaction center nodes
                smiles = self.get_reaction_nodes_label(ec, reactant, reaction_nodes)

                for uniprot in valid_uniprots:
                    sample = {
                        "smiles": smiles,
                        "ec": ec,
                        "rowid": f"{uniprot}_{reaction['rxnid']}_{reactant_id}",
                        "sample_id": f"{uniprot}_{reaction['rxnid']}_{reactant_id}",
                        "uniprot_id": uniprot,
                        "protein_id": uniprot,
                        "split": reaction["split"],
                        "y": 1,
                        "organism": reaction.get("organism", ""),
                    }

                    for ec_level, _ in enumerate(ec.split(".")):
                        sample[f"ec{ec_level+1}"] = ".".join(
                            ec.split(".")[: (ec_level + 1)]
                        )

                    # remove duplicate prot-substrate pairs
                    if f"{uniprot}_{smiles}" in seen_before:
                        continue
                    seen_before.add(f"{uniprot}_{smiles}")
                    # add reaction sample to dataset
                    try:
                        if self.args.use_protein_graphs:
                            graph_path = os.path.join(
                                self.args.protein_graphs_dir,
                                "processed",
                                f"{sample['uniprot_id']}_graph.pt",
                            )

                            if not os.path.exists(graph_path):
                                data = self.create_protein_graph(sample)
                                if data is None:
                                    raise Exception("Could not generate protein graph")
                                torch.save(data, graph_path)
                        dataset.append(sample)

                    except Exception as e:
                        print(f"Error processing {sample['sample_id']} because of {e}")
                        continue
        return dataset

    def get_reaction_nodes_label(self, ec, mapped_reactant, reaction_nodes):
        """get binary vector of all nodes participating in reaction"""
        mol = Chem.MolFromSmiles(mapped_reactant)
        atom_maps = {atom.GetAtomMapNum(): i for i, atom in enumerate(mol.GetAtoms())}
        [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]
        smiles = Chem.MolToSmiles(mol)
        if not len(self.ec_substrate2reaction_center.get(f"{ec}_{smiles}", [])):
            self.ec_substrate2reaction_center[f"{ec}_{smiles}"] = torch.zeros(
                mol.GetNumAtoms()
            )
        for j in reaction_nodes:
            if j in atom_maps:
                self.ec_substrate2reaction_center[f"{ec}_{smiles}"][atom_maps[j]] = 1
        return smiles

    def __getitem__(self, index):
        item = super().__getitem__(index)

        sample = self.dataset[index]
        ec = sample["ec"]
        smiles = sample["smiles"]

        if isinstance(item, Data):
            item.reaction_nodes = self.ec_substrate2reaction_center[f"{ec}_{smiles}"]
            item = {
                "mol": item,
                "sequence": item.sequence,
                "sequence_annotation": item.sequence_annotation,
                "sample_id": item.sample_id,
                "ec": ec,
            }
        else:
            raise NotImplementedError

        return item

    @staticmethod
    def set_args(args):
        pass


@register_object("enzyme_map+uspto_graph", "dataset")
class EMap_USPTOGraph(EnzymeMapGraph):
    def __init__(self, args, split_group) -> None:
        super().__init__(args, split_group)

        emap_dataset = self.dataset
        cargs = copy.deepcopy(args)
        cargs.dataset_file_path = args.uspto_dataset_file_path
        if split_group == "train":
            uspto_dataset = get_object("chemical_reactions_graph", "dataset")(
                cargs, split_group
            ).dataset
        else:
            uspto_dataset = []

        for d in uspto_dataset:
            d["dataset"] = "uspto"
        for d in emap_dataset:
            d["dataset"] = "emap"

        self.dataset = emap_dataset + uspto_dataset
        self.set_sample_weights(args)
        self.print_summary_statement(self.dataset, split_group)

    @staticmethod
    def add_args(parser):
        EnzymeMap.add_args(parser)
        parser.add_argument(
            "--uspto_dataset_file_path", default=None, help="path to file path"
        )
        parser.add_argument(
            "--emap_dataset_file_path", default=None, help="path to file path"
        )


@register_object("enzyme_map+uspto", "dataset")
class EMap_USPTO(EnzymeMap):
    def __init__(self, args, split_group) -> None:
        super().__init__(args, split_group)

        emap_dataset = self.dataset
        cargs = copy.deepcopy(args)
        cargs.dataset_file_path = args.uspto_dataset_file_path
        if split_group == "train":
            uspto_dataset = get_object("chemical_reactions", "dataset")(
                cargs, split_group
            ).dataset
        else:
            uspto_dataset = []

        for d in uspto_dataset:
            d["dataset"] = "uspto"
        for d in emap_dataset:
            d["dataset"] = "emap"

        if split_group == "train":
            self.dataset = emap_dataset + uspto_dataset
        else:
            self.dataset = emap_dataset

        reaction_to_products = defaultdict(set)
        for sample in uspto_dataset:
            key = f"{sample['ec']}{'.'.join(sample['reactants'])}"
            reaction_to_products[key].add(".".join(sample["products"]))
        self.reaction_to_products.update(reaction_to_products)

        self.dataset = self.dataset
        self.set_sample_weights(args)
        self.print_summary_statement(self.dataset, split_group)

    @staticmethod
    def add_args(parser):
        EnzymeMap.add_args(parser)
        parser.add_argument(
            "--uspto_dataset_file_path", default=None, help="path to file path"
        )
        parser.add_argument(
            "--emap_dataset_file_path", default=None, help="path to file path"
        )


@register_object("enzyme_map+uspto_val", "dataset")
class EMap_USPTO_Val(EMap_USPTO):
    def __init__(self, args, split_group) -> None:
        split = "dev" if split_group == "train" else split_group
        super().__init__(args, split)
