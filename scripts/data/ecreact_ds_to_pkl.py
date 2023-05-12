import sys, os

sys.path.append(os.path.dirname(os.path.realpath(".")))
import pickle
from argparse import Namespace
from nox.utils.registry import get_object
from nox.utils.loading import get_lightning_model, get_eval_dataset_loader
from tqdm import tqdm

args = Namespace(
    **pickle.load(
        open(
            "/Mounts/rbg-storage1/logs/metabo/3f77fda5cce7cbc64e8a0ad7fc74753f.args",
            "rb",
        )
    )
)
args.batch_size = 1
args.world_size = 1
args.num_processes = 1
args.gpus = 1
args.sample_negatives_range = [0.6, 1.0]
train_data = get_object(args.dataset_name, "dataset")(args, "train")
val_data = get_object(args.dataset_name, "dataset")(args, "dev")
test_data = get_object(args.dataset_name, "dataset")(args, "test")
dataset = []
for data in [train_data, val_data, test_data]:
    for s in tqdm(data):
        dataset.append(s)

pickle.dump(
    dataset,
    open("/Mounts/rbg-storage1/datasets/Enzymes/ecreact_with_negatives.json", "wb"),
)
