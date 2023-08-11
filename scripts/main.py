import sys, os
import rdkit

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
os.environ["WANDB__SERVICE_WAIT"] = "300"
from ast import arg
from collections import OrderedDict
import pickle
import time
import git
import copy
import comet_ml
import torch
import pytorch_lightning as pl
from pytorch_lightning import _logger as log
import inspect
import torch_geometric

from nox.utils.parsing import parse_args
from nox.utils.registry import get_object
import nox.utils.loading as loaders
from nox.utils.callbacks import set_callbacks
from rich import print

from pytorch_lightning.strategies import DDPStrategy

def train(args):
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
        trainer_args = {k: cast_type(v) for k, v in vars(args).items() if k in trainer_arg_names}
        if int(args.devices) > 1:
            trainer_args["strategy"] = DDPStrategy(find_unused_parameters=True)
            args.strategy = 'ddp' # important for loading
        else:
            trainer_args["strategy"] = 'auto'
            args.strategy = 'auto'
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

    repo = git.Repo(search_parent_directories=True)
    commit = repo.head.object
    log.info(
        "\nProject main running by author: {} \ndate:{}, \nfrom commit: {} -- {}".format(
            commit.author,
            time.strftime("%m-%d-%Y %H:%M:%S", time.localtime(commit.committed_date)),
            commit.hexsha,
            commit.message,
        )
    )

    train_dataset = loaders.get_train_dataset_loader(args)
    dev_dataset = loaders.get_eval_dataset_loader(args, split="dev")

    # print args
    for key, value in sorted(vars(args).items()):
        print("{} -- {}".format(key.upper(), value))

    # create or load lightning model from checkpoint
    model = loaders.get_lightning_model(args)
    # compile model
    # model = torch.compile(model)
    # model.model = torch_geometric.compile(model.model)

    # logger
    trainer.logger = get_object(args.logger_name, "logger")(args)

    # push to logger
    trainer.logger.setup(**{"args": args, "model": model})

    # add callbacks
    trainer.callbacks = set_callbacks(trainer, args)

    # train model
    if args.train:
        log.info("\nTraining Phase...")
        trainer.fit(model, train_dataset, dev_dataset)
        if trainer.checkpoint_callback:
            args.model_path = trainer.checkpoint_callback.best_model_path

    # save args
    if args.local_rank == 0:
        print("Saving args to {}.args".format(args.results_path))
        pickle.dump(vars(args), open("{}.args".format(args.results_path), "wb"))

    return model, trainer.logger


def eval(model, logger, args):
    # reinit trainer
    if not hasattr(pl.Trainer, "add_argparse_args"):
        trainer = pl.Trainer(devices=1)
    else:
        trainer = pl.Trainer(gpus=1)

    # change model args
    model.args.num_nodes = trainer.num_nodes
    model.args.num_processes = trainer.num_devices
    model.args.world_size = trainer.num_nodes * trainer.num_devices
    model.args.global_rank = trainer.global_rank
    model.args.local_rank = trainer.local_rank

    # reset ddp
    # just keeps track for args, trainer is reinitialized above
    if not hasattr(pl.Trainer, "add_argparse_args"):
        args.devices = 1
        args.strategy = "auto" # note must set devices=1 otherwise will use all available gpus
    else:
        args.strategy = None # legacy

    # connect to same logger as in training
    trainer.logger = logger

    # set callbacks
    trainer.callbacks = set_callbacks(trainer, args)

    # eval on train
    if args.eval_on_train:
        log.info("\nInference Phase on train set...")
        train_dataset = loaders.get_eval_dataset_loader(args, split="train")

        if args.train and trainer.checkpoint_callback:
            trainer.test(model, train_dataset, ckpt_path=args.model_path)
        else:
            trainer.test(model, train_dataset)

    # eval on dev
    if args.dev:
        log.info("\nValidation Phase...")
        dev_dataset = loaders.get_eval_dataset_loader(args, split="dev")
        if args.train and trainer.checkpoint_callback:
            trainer.test(model, dev_dataset, ckpt_path=args.model_path)
        else:
            trainer.test(model, dev_dataset)

    # eval on test
    if args.test:
        log.info("\nInference Phase on test set...")
        test_dataset = loaders.get_eval_dataset_loader(args, split="test")

        if args.train and trainer.checkpoint_callback:
            trainer.test(model, test_dataset, ckpt_path=args.model_path)
        else:
            trainer.test(model, test_dataset)


if __name__ == "__main__":
    args = parse_args()
    model, logger = train(args)

    if args.dev or args.test or args.eval_on_train:
        if args.strategy == "ddp":
            torch.distributed.destroy_process_group()
            log.info("\n\n")
            log.info(">" * 33)
            log.info("Destroyed process groups for eval")
            log.info("<" * 33)
            log.info("\n\n")

        if args.global_rank == 0:
            eval(model, logger, args)
