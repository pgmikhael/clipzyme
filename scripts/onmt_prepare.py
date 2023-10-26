import rdkit
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import subprocess
import shutil
import pickle
from nox.utils.parsing import parse_args
from nox.utils.registry import get_object
from rich import print
from rdkit import Chem
from rxn.chemutils.utils import remove_atom_mapping
from tqdm import tqdm
import json

rbt_dir = "/Mounts/rbg-storage1/users/pgmikhael/rbt-preprocess"


def write_kreuter_rxns(df, directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    for split, rows in tqdm(rowid_to_split.items(), ncols=100):
        with open(os.path.join(directory, f"src-{split}.txt"), "w") as f, open(
            os.path.join(directory, f"tgt-{split}.txt"), "w"
        ) as g:
            for rowid in rows:
                f.write(df[rowid]["TransformerIn"])
                f.write("\n")

                g.write(df[rowid]["TransformerOut"])
                g.write("\n")


def get_reactions(dataset):
    rxns = set()
    for sample in tqdm(dataset, ncols=100, leave=False):
        rs = [Chem.CanonSmiles(remove_atom_mapping(r)) for r in sample["reactants"]]
        ps = [Chem.CanonSmiles(remove_atom_mapping(r)) for r in sample["products"]]
        reaction = "{}|{}>>{}".format(".".join(rs), sample["ec"], ps[0])
        rxns.add(reaction)
    return rxns


def write_probst_reactions(rxns, directory, path):
    if not os.path.isdir(directory):
        os.makedirs(directory)

    with open(os.path.join(directory, f"{path}.txt"), "w") as f:
        for reaction in rxns:
            f.write(reaction)
            f.write("\n")


def rbt_process_data(split, experiment_name):
    return f"""python {rbt_dir}/bin/rbt-preprocess-single-file.py \
        {rbt_dir}/datasets/{experiment_name}/{split}.txt \
        {rbt_dir}/datasets/{experiment_name}-process \
        --remove-patterns {rbt_dir}/data/patterns.txt \
        --remove-molecules {rbt_dir}/data/molecules.txt \
        --ec-level 3 \
        --min-atom-count 4"""


def cleanup_rbt_process_output(split, experiment_name):
    directory = f"{rbt_dir}/datasets/{experiment_name}-process/experiments/3"
    shutil.move(
        f"{directory}/src.txt",
        f"{rbt_dir}/datasets/{experiment_name}/src-{split}.txt",
    )
    shutil.move(
        f"{directory}/tgt.txt",
        f"{rbt_dir}/datasets/{experiment_name}/tgt-{split}.txt",
    )

    shutil.rmtree(f"{rbt_dir}/datasets/{experiment_name}-process/")


def remove_ec_from_src_tgt_files(from_path, to_path):
    if not os.path.isdir(to_path):
        os.makedirs(to_path)

    for split in ["train", "valid", "test"]:
        with open(f"{from_path}/src-{split}.txt", "r") as f, open(
            f"{to_path}/src-{split}.txt", "w"
        ) as g:
            for line in f:
                line_ = line.split("|")[0]
                g.write(line_)
                g.write("\n")

        with open(f"{from_path}/tgt-{split}.txt", "r") as f, open(
            f"{to_path}/tgt-{split}.txt", "w"
        ) as g:
            for line in f:
                g.write(line)


def make_experiment_name(args):
    if args.split_type == "mmseqs":
        split = (
            "foldseek_"
            + os.path.splitext(os.path.basename(args.uniprot2cluster_path))[0]
        )
    elif args.split_type == "ec_hold_out":
        split = "ec_" + str(args.held_out_ec_num)
    else:
        raise NotImplementedError
    experiment_name = f"{split}_split_with_ec"
    return experiment_name


if __name__ == "__main__":
    args = parse_args()

    train_data = get_object(args.dataset_name, "dataset")(args, "train")
    dev_data = get_object(args.dataset_name, "dataset")(args, "dev")
    test_data = get_object(args.dataset_name, "dataset")(args, "test")

    args.experiment_name = make_experiment_name(args)

    # -----------------------------------------------------------
    # Probst Data
    # -----------------------------------------------------------

    rxns = get_reactions(train_data.dataset)
    write_probst_reactions(
        rxns,
        f"{rbt_dir}/datasets/{args.experiment_name}",
        "train",
    )

    rxns = get_reactions(dev_data.dataset)
    write_probst_reactions(
        rxns,
        f"{rbt_dir}/datasets/{args.experiment_name}",
        "valid",
    )

    rxns = get_reactions(test_data.dataset)
    write_probst_reactions(
        rxns,
        f"{rbt_dir}/datasets/{args.experiment_name}",
        "test",
    )
    for split in ["train", "valid", "test"]:
        os.makedirs(f"{rbt_dir}/datasets/{args.experiment_name}-process")
        subprocess.call(rbt_process_data(split, args.experiment_name), shell=True)
        cleanup_rbt_process_output(split, args.experiment_name)

    remove_ec_from_src_tgt_files(
        from_path=f"{rbt_dir}/datasets/{args.experiment_name}",
        to_path=f"{rbt_dir}/datasets/{args.experiment_name}".replace(
            "with_ec", "without_ec"
        ),
    )

    # -----------------------------------------------------------
    # Kreuter Data
    # -----------------------------------------------------------
    kreuter_data = json.load(
        open(
            "/Mounts/rbg-storage1/users/pgmikhael/notebooks/FwdSynthesis/kreuter_emap2_dataset.json",
            "r",
        )
    )

    rowid_to_split = {}
    for data, split in [
        (train_data, "train"),
        (dev_data, "valid"),
        (test_data, "test"),
    ]:
        rowid_to_split[split] = [d["df_row"] for d in data.dataset]

    write_kreuter_rxns(
        kreuter_data,
        f"{rbt_dir}/datasets/kreutter_{args.experiment_name[:-len('_with_ec')]}",
    )
