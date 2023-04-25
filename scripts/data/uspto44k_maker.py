import json
import os 

mapper_dataset_files = [
    ("train","/Mounts/rbg-storage1/datasets/ChemicalReactions/uspto44k_wengong/train.txt"),
    ("dev", "/Mounts/rbg-storage1/datasets/ChemicalReactions/uspto44k_wengong/valid.txt"),
    ("test", "/Mounts/rbg-storage1/datasets/ChemicalReactions/uspto44k_wengong/test.txt"),
]

if __name__ == "__main__":
    dataset = []
    for split, filepath in mapper_dataset_files:
        if filepath.endswith(".txt"):
            with open(filepath, "r") as f:
                line_counter = 0
                for rxn in f:
                    dataset.append(
                        {
                            "reaction": rxn.rstrip("\n"),
                            "split": split,
                            "from": filepath,
                            "rxnid": line_counter,
                        }
                    )
                    line_counter += 1
        else:
            raise NotImplementedError

    json.dump(
        dataset,
        open(
            "/Mounts/rbg-storage1/datasets/ChemicalReactions/uspto44k_chem_reactions_dataset.json",
            "w",
        ),
    )