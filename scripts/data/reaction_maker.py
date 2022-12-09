import json
import os 

mapper_dataset_files = [
    (
        "train",
        "/Mounts/rbg-storage1/datasets/ChemicalReactions/mapper/uspto_all_reactions_training.txt",
    ),
    ("dev", "/Mounts/rbg-storage1/datasets/ChemicalReactions/mapper/eval_schneider.json"),
    ("dev", "/Mounts/rbg-storage1/datasets/ChemicalReactions/mapper/test_schneider.json"),
    ("test", "/Mounts/rbg-storage1/datasets/ChemicalReactions/mapper/test_natcomm.json"),
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

    # synthesis
    parent_dir = "/Mounts/rbg-storage1/datasets/ChemicalReactions/synthesis"
    forward_synthesis_dirs = os.listdir(parent_dir)
    for directory in forward_synthesis_dirs:
        files = os.listdir(os.path.join(parent_dir, directory))

        formatted_dataset = {
            "train": {"src": [], "tgt": []},
            "val": {"src": [], "tgt": []},
            "test": {"src": [], "tgt": []},
        }
        
        for f in files:
            rxn_side, split = f.split('-')
            split, _ = os.path.splitext(split)
            filepath = os.path.join(parent_dir, directory, f)
            with open(filepath, "r") as f:
                for line in f:
                    formatted_dataset[split][rxn_side].append(line.rstrip("\n").replace(" ", ""))

        dataset = []
        for split, data_items in formatted_dataset.items():
            assert len(data_items["src"]) == len(data_items["tgt"])
            for i, (src, tgt) in enumerate(zip(data_items["src"], data_items["tgt"])):
                dataset.append({
                    "reaction": "{}>>{}".format(src, tgt),
                    "split": "dev" if split == "val" else split,
                    "from": filepath,
                    "rxnid": f"{split}_{i}",
                })
        
        json.dump(
            dataset,
            open(
                f"/Mounts/rbg-storage1/datasets/ChemicalReactions/{directory}_synthesis_dataset.json",
                "w",
            ),
        )




