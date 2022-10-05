import json

dataset_files = [
    (
        "train",
        "/Mounts/rbg-storage1/datasets/ChemicalReactions/uspto_all_reactions_training.txt",
    ),
    ("val", "/Mounts/rbg-storage1/datasets/ChemicalReactions/eval_schneider.json"),
    ("val", "/Mounts/rbg-storage1/datasets/ChemicalReactions/test_schneider.json"),
    ("test", "/Mounts/rbg-storage1/datasets/ChemicalReactions/test_natcomm.json"),
]

if __name__ == "__main__":
    dataset = []
    for split, filepath in dataset_files:
        if filepath.endswith(".txt"):
            with open(filepath, "r") as f:
                line_counter = 0
                for rxn in f:
                    dataset.append(
                        {
                            "reaction": rxn,
                            "split": split,
                            "from": filepath,
                            "rxnid": line_counter,
                        }
                    )
                    line_counter += 1
        elif filepath.endswith(".json"):
            jsonfile = json.load(open(filepath, "r"))
            jsonfile = jsonfile["rxn"]
            for rxnid, rxn in jsonfile.items():
                dataset.append(
                    {"reaction": rxn, "split": split, "rxnid": rxnid, "from": filepath}
                )
        else:
            raise NotImplementedError

    json.dump(
        dataset,
        open(
            "/Mounts/rbg-storage1/datasets/ChemicalReactions/chem_reactions_dataset.json",
            "w",
        ),
    )
