import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

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

from nox.utils.parsing import parse_args
from nox.utils.registry import get_object
import nox.utils.loading as loaders
from nox.utils.callbacks import set_callbacks


def train(args):

    # Remove callbacks from args for safe pickling later
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
    trainer = pl.Trainer(gpus=1)

    # reset ddp
    args.strategy = None

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
