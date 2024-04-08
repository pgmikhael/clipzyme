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
    if hasattr(args, "gpus") and not hasattr(pl.Trainer, "add_argparse_args"):
        args.devices = args.gpus

    # using gpus
    if (isinstance(args.gpus, str) and len(args.gpus.split(",")) > 1) or (
        isinstance(args.gpus, int) and args.gpus > 1
    ):
        args.strategy = "ddp"
        # should not matter since we set our sampler, in 2.0 this is default True and used to be called replace_sampler_ddp
        args.use_distributed_sampler = False
    else:
        if not hasattr(pl.Trainer, "add_argparse_args"):  # lightning 2.0
            # args.strategy = "auto"
            args.strategy = "ddp"  # should be overwritten later in main
        else:  # legacy
            args.strategy = None
            args.replace_sampler_ddp = False

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
        if int(args.gpus) > 1:
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

    # hard code dataset name
    args.dataset_name = "reactions_dataset"
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
        "--precision",
        default="bf16",
        help="precision to use for eval",
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
        help="Filename of model snapshot to load [default: None]",
    )
    parser.add_argument(
        "--use_as_protein_encoder",
        action="store_true",
        default=False,
        help="Use the model as a protein encoder [default: False]",
    )
    parser.add_argument(
        "--use_as_reaction_encoder",
        action="store_true",
        default=False,
        help="Use the model as a reaction encoder [default: False]",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Number of GPUs to train on",
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
        default="logs/test",
        help="Where to save the arguments of the run",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="defined either automatically by dispatcher.py or time in main.py. Keep without default",
    )

    args = parser.parse_args()
    eval(args)
