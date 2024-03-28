import rdkit
import pickle
import torch
import pytorch_lightning as pl
from pytorch_lightning import _logger as log
import inspect
import argparse
import clipzyme.utils.loading as loaders
from clipzyme.lightning.clipzyme import CLIPZyme
from rich import print

from pytorch_lightning.strategies import DDPStrategy

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
torch.multiprocessing.set_sharing_strategy("file_system")


def eval(args):
    # legacy
    if not hasattr(pl.Trainer, "from_argparse_args"):

        def cast_type(val):
            if isinstance(val, str):
                if val.isdigit():
                    return int(val)
                try:
                    return float(val)
                except ValueError:
                    return val
            return val

        # Get the trainer's argument names
        trainer_arg_names = set(inspect.signature(pl.Trainer).parameters.keys())
        trainer_args = {
            k: cast_type(v) for k, v in vars(args).items() if k in trainer_arg_names
        }
        if int(args.devices) > 1:
            trainer_args["strategy"] = DDPStrategy(find_unused_parameters=True)
            args.strategy = "ddp"  # important for loading
        else:
            trainer_args["strategy"] = "auto"
            args.strategy = "auto"
        trainer = pl.Trainer(**trainer_args)
    else:
        # Remove callbacks from args for safe pickling later
        args.find_unused_parameters = False
        trainer = pl.Trainer.from_argparse_args(args)
    args.callbacks = None
    args.num_nodes = trainer.num_nodes
    args.num_processes = trainer.num_devices
    args.world_size = args.num_nodes * args.num_processes
    args.global_rank = trainer.global_rank
    args.local_rank = trainer.local_rank

    dataset = loaders.get_eval_dataset_loader(args, split="test", shuffle=False)

    # print args
    for key, value in sorted(vars(args).items()):
        print("{} -- {}".format(key.upper(), value))

    # create or load lightning model from checkpoint
    model = CLIPZyme(args)

    # run screening
    log.info("\nScreening in progress...")
    trainer.test(model, dataset)

    # save args
    if args.local_rank == 0:
        print("Saving args to {}.args".format(args.results_path))
        pickle.dump(vars(args), open("{}.args".format(args.results_path), "wb"))

    return model, trainer.logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Nox Standard Args.", allow_abbrev=False
    )
    # dataset args
    parser.add_argument(
        "--dataset_file_path",
        type=str,
        default=None,
        help="Path to dataset file",
    )
    parser.add_argument(
        "--esm_dir",
        type=str,
        default="/home/ubuntu/esm/esm2_t33_650M_UR50D.pt",
        help="directory to load esm model from",
    )
    parser.add_argument(
        "--protein_cache_dir",
        type=str,
        default=None,
        help="directory to save load load protein graphs from",
    )
    # loading args
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training [default: 128]",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Num workers for each data loader [default: 4]",
    )
    # model args
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Filename of model snapshot to load[default: None]",
    )
    # results args
    parser.add_argument(
        "--save_hiddens",
        action="store_true",
        default=False,
        help="Save hidden repr from each image to an npz based off results path, git hash and exam name",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        default=False,
        help="Save hidden repr from each image to an npz based off results path, git hash and exam name",
    )
    parser.add_argument(
        "--inference_dir",
        type=str,
        default="hiddens/test_run",
        help='Dir to store hiddens npy"s when store_hiddens is true',
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="logs/test.args",
        help="Where to save the result logs",
    )

    args = parser.parse_args()
    eval(args)
