import argparse
import os
import pwd
from pytorch_lightning import Trainer
import itertools
from clipzyme.utils.registry import md5
import json
import copy
from clipzyme.utils.classes import set_nox_type
import inspect

EMPTY_NAME_ERR = 'Name of augmentation or one of its arguments cant be empty\n\
                  Use "name/arg1=value/arg2=value" format'
POSS_VAL_NOT_LIST = (
    "Flag {} has an invalid list of values: {}. Length of list must be >=1"
)


class GlobalNamespace(argparse.Namespace):
    pass


def parse_dispatcher_config(config):
    """
    Parses an experiment config, and creates jobs. For flags that are expected to be a single item,
    but the config contains a list, this will return one job for each item in the list.
    :config - experiment_config

    returns: jobs - a list of flag strings, each of which encapsulates one job.
         *Example: --train --cuda --dropout=0.1 ...
    returns: experiment_axies - axies that the grid search is searching over
    """

    assert all(
        [
            k
            in [
                "script",
                "available_gpus",
                "cartesian_hyperparams",
                "paired_hyperparams",
                "tune_hyperparams",
            ]
            for k in config.keys()
        ]
    )

    cartesian_hyperparamss = config["cartesian_hyperparams"]
    paired_hyperparams = config.get("paired_hyperparams", [])
    flags = []
    arguments = []
    experiment_axies = []

    # add anything outside search space as fixed
    fixed_args = ""
    for arg in config:
        if arg not in [
            "script",
            "cartesian_hyperparams",
            "paired_hyperparams",
            "available_gpus",
        ]:
            if type(config[arg]) is bool:
                if config[arg]:
                    fixed_args += "--{} ".format(str(arg))
                else:
                    continue
            else:
                fixed_args += "--{} {} ".format(arg, config[arg])

    # add paired combo of search space
    paired_args_list = [""]
    if len(paired_hyperparams) > 0:
        paired_args_list = []
        paired_keys = list(paired_hyperparams.keys())
        paired_vals = list(paired_hyperparams.values())
        flags.extend(paired_keys)
        for paired_combo in zip(*paired_vals):
            paired_args = ""
            for i, flg_value in enumerate(paired_combo):
                if type(flg_value) is bool:
                    if flg_value:
                        paired_args += "--{} ".format(str(paired_keys[i]))
                    else:
                        continue
                else:
                    paired_args += "--{} {} ".format(
                        str(paired_keys[i]), str(flg_value)
                    )
            paired_args_list.append(paired_args)

    # add every combo of search space
    product_flags = []
    for key, value in cartesian_hyperparamss.items():
        flags.append(key)
        product_flags.append(key)
        arguments.append(value)
        if len(value) > 1:
            experiment_axies.append(key)

    experiments = []
    exps_combs = list(itertools.product(*arguments))

    for tpl in exps_combs:
        exp = ""
        for idx, flg in enumerate(product_flags):
            if type(tpl[idx]) is bool:
                if tpl[idx]:
                    exp += "--{} ".format(str(flg))
                else:
                    continue
            else:
                exp += "--{} {} ".format(str(flg), str(tpl[idx]))
        exp += fixed_args
        for paired_args in paired_args_list:
            experiments.append(exp + paired_args)

    return experiments, flags, experiment_axies


def prepare_training_config_for_eval(train_config):
    """Convert training config to an eval config for testing.

    Parameters
    ----------
    train_config: dict
         config with the following structure:
              {
                   "train_config": ,   # path to train config
                   "log_dir": ,        # log directory used by dispatcher during training
                   "eval_args": {}     # test set-specific arguments beyond default
              }

    Returns
    -------
    experiments: list
    flags: list
    experiment_axies: list
    """

    train_args = json.load(open(train_config["train_config"], "r"))

    experiments, _, _ = parse_dispatcher_config(train_args)
    stem_names = [md5(e) for e in experiments]
    eval_args = copy.deepcopy(train_args)
    eval_args["cartesian_hyperparams"].update(train_config["eval_args"])

    # reset defaults
    eval_args["cartesian_hyperparams"]["train"] = [False]
    eval_args["cartesian_hyperparams"]["test"] = [True]
    eval_args["cartesian_hyperparams"]["from_checkpoint"] = train_config[
        "eval_args"
    ].get("from_checkpoint", [True])
    eval_args["cartesian_hyperparams"]["gpus"] = [1]
    eval_args["cartesian_hyperparams"]["logger_tags"][0] += " eval"
    eval_args["available_gpus"] = train_config["available_gpus"]
    eval_args["script"] = train_config["script"]

    experiments, flags, experiment_axies = parse_dispatcher_config(eval_args)

    if "checkpoint_path" not in eval_args["cartesian_hyperparams"]:
        for (idx, e), s in zip(enumerate(experiments), stem_names):
            experiments[idx] += " --checkpoint_path {}".format(
                os.path.join(train_config["log_dir"], "{}.args".format(s))
            )

    return experiments, flags, experiment_axies


def parse_augmentations(augmentations):
    """
    Parse the list of augmentations, given by configuration, into a list of
    tuple of the augmentations name and a dictionary containing additional args.

    The augmentation is assumed to be of the form 'name/arg1=value/arg2=value'

    :raw_augmentations: list of strings [unparsed augmentations]
    :returns: list of parsed augmentations [list of (name,additional_args)]

    """
    raw_transformers = augmentations

    transformers = []
    for t in raw_transformers:
        arguments = t.split("/")
        name = arguments[0]
        if name == "":
            raise Exception(EMPTY_NAME_ERR)

        kwargs = {}
        if len(arguments) > 1:
            for a in arguments[1:]:
                splited = a.split("=")
                var = splited[0]
                val = splited[1] if len(splited) > 1 else None
                if var == "":
                    raise Exception(EMPTY_NAME_ERR)
                try:
                    kwargs[var] = float(val)
                except ValueError:
                    kwargs[var] = val

        transformers.append((name, kwargs))

    return transformers


def get_parser():
    global_namespace = GlobalNamespace(allow_abbrev=False)

    parser = argparse.ArgumentParser(
        description="Nox Standard Args.", allow_abbrev=False
    )

    # -------------------------------------
    # Run Setup
    # -------------------------------------
    parser.add_argument(
        "--train",
        action="store_true",
        default=False,
        help="Whether or not to train model",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        default=False,
        help="Whether or not to run model on dev set",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="Whether or not to run model on test set",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        default=False,
        help="Whether to run model for pure prediction where labels are not known",
    )
    parser.add_argument(
        "--eval_on_train",
        action="store_true",
        default=False,
        help="Whether or not to evaluate model on train split",
    )
    parser.add_argument(
        "--eval_on_train_multigpu",
        action="store_true",
        default=False,
        help="Whether or not to evaluate model on train split using ddp",
    )
    parser.add_argument(
        "--replicate",
        type=int,
        default=1,
        help="The replicate number for the experiment for running same experiments multiple times",
    )
    parser.add_argument(
        "--shuffle_eval_loader",
        action="store_true",
        default=False,
        help="shuffle the dev and test datasets",
    )

    # -------------------------------------
    # Data
    # -------------------------------------
    parser.add_argument(
        "--dataset_name",
        type=str,
        action=set_nox_type("dataset"),
        default="mnist",
        help="Name of dataset",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        nargs="+",
        default=[256, 256],
        help="Width and height of image in pixels. [default: [256,256]]",
    )
    parser.add_argument(
        "--num_chan", type=int, default=3, help="Number of channels for input image"
    )
    parser.add_argument(
        "--img_mean",
        type=float,
        nargs="+",
        default=[128.1722],
        help="Mean of image per channel",
    )
    parser.add_argument(
        "--img_std",
        type=float,
        nargs="+",
        default=[87.1849],
        help="Standard deviation  of image per channel",
    )
    parser.add_argument(
        "--img_file_type",
        type=str,
        default="png",
        choices=["png", "dicom"],
        help="Type of image. one of [png, dicom]",
    )

    # -------------------------------------
    # Augmentations
    # -------------------------------------
    parser.add_argument(
        "--train_rawinput_augmentation_names",
        nargs="*",
        action=set_nox_type("augmentation"),
        default=[],
        help='List of image-transformations to use. Usage: "--train_rawinput_augmentations trans1/arg1=5/arg2=2 trans2 trans3/arg4=val"',
    )
    parser.add_argument(
        "--train_tnsr_augmentation_names",
        nargs="*",
        action=set_nox_type("augmentation"),
        default=[],
        help='List of image-transformations to use. Usage: "--train_tnsr_augmentations trans1/arg1=5/arg2=2 trans2 trans3/arg4=val"',
    )
    parser.add_argument(
        "--test_rawinput_augmentation_names",
        nargs="*",
        action=set_nox_type("augmentation"),
        default=[],
        help="List of image-transformations to use for the dev and test dataset",
    )
    parser.add_argument(
        "--test_tnsr_augmentation_names",
        nargs="*",
        action=set_nox_type("augmentation"),
        default=[],
        help="List of image-transformations to use for the dev and test dataset",
    )

    # -------------------------------------
    # Losses
    # -------------------------------------

    # losses and metrics
    parser.add_argument(
        "--loss_names",
        type=str,
        action=set_nox_type("loss"),
        nargs="*",
        default=[],
        help="Name of loss",
    )
    parser.add_argument(
        "--loss_names_for_eval",
        type=str,
        action=set_nox_type("loss"),
        nargs="*",
        default=None,
        help="Name of loss",
    )

    # -------------------------------------
    # Metrics
    # -------------------------------------

    parser.add_argument(
        "--metric_names",
        type=str,
        action=set_nox_type("metric"),
        nargs="*",
        default=[],
        help="Name of performance metric",
    )
    parser.add_argument(
        "--metric_names_for_eval",
        type=str,
        action=set_nox_type("metric"),
        nargs="*",
        default=None,
        help="Name of metric",
    )

    # -------------------------------------
    # Training Module
    # -------------------------------------

    parser.add_argument(
        "--lightning_name",
        type=str,
        action=set_nox_type("lightning"),
        default="base",
        help="Name of lightning module",
    )

    # -------------------------------------
    # Hyper parameters
    # -------------------------------------
    # learning
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training [default: 128]",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Initial learning rate [default: 0.001]",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.25,
        help="Amount of dropout to apply on last hidden layer [default: 0.25]",
    )
    parser.add_argument(
        "--optimizer_name",
        type=str,
        action=set_nox_type("optimizer"),
        default="adam",
        help="Optimizer to use [default: adam]",
    )
    parser.add_argument(
        "--momentum", type=float, default=0, help="Momentum to use with SGD"
    )
    parser.add_argument(
        "--lr_decay",
        type=float,
        default=0.1,
        help="Initial learning rate [default: 0.5]",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0,
        help="L2 Regularization penaty [default: 0]",
    )

    # tune
    parser.add_argument(
        "--tune_hyperopt",
        action="store_true",
        default=False,
        help="Whether to run hyper-parameter optimization",
    )
    parser.add_argument(
        "--tune_search_alg",
        type=str,
        default="search",
        help="Optimization algorithm",
    )
    parser.add_argument(
        "--tune_hyperparam_names",
        type=str,
        nargs="*",
        default=[],
        help="Name of parameters being optimized",
    )

    # -------------------------------------
    # Schedule
    # -------------------------------------
    parser.add_argument(
        "--scheduler_name",
        type=str,
        action=set_nox_type("scheduler"),
        default="reduce_on_plateau",
        help="Name of scheduler",
    )
    parser.add_argument(
        "--cosine_annealing_period",
        type=int,
        default=10,
        help="length of period of lr cosine anneal",
    )
    parser.add_argument(
        "--cosine_annealing_period_scaling",
        type=int,
        default=2,
        help="how much to multiply each period in successive annealing",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Number of epochs without improvement on dev before halving learning rate and reloading best model [default: 5]",
    )
    parser.add_argument(
        "--num_adv_steps",
        type=int,
        default=1,
        help="Number of steps for domain adaptation discriminator per one step of encoding model [default: 5]",
    )

    # -------------------------------------
    # Callbacks
    # -------------------------------------

    parser.add_argument(
        "--callback_names",
        type=str,
        action=set_nox_type("callback"),
        nargs="*",
        default=["checkpointer", "lr_monitor"],
        help="Lightning callbacks",
    )

    parser.add_argument(
        "--monitor",
        type=str,
        default=None,
        help="Name of metric to use to decide when to save model",
    )

    parser.add_argument(
        "--checkpoint_save_top_k",
        type=int,
        default=1,
        help="the best k models according to the quantity monitored will be saved",
    )
    parser.add_argument(
        "--checkpoint_save_last",
        action="store_true",
        default=False,
        help="save the last model to last.ckpt",
    )

    # -------------------------------------
    # Model checkpointing
    # -------------------------------------

    parser.add_argument(
        "--checkpoint_dir", type=str, default="snapshot", help="Where to dump the model"
    )
    parser.add_argument(
        "--from_checkpoint",
        action="store_true",
        default=False,
        help="Whether loading a model from a saved checkpoint",
    )
    parser.add_argument(
        "--relax_checkpoint_matching",
        action="store_true",
        default=False,
        help="Do not enforce that the keys in checkpoint_path match the keys returned by this module’s state dict",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Filename of model snapshot to load[default: None]",
    )

    # -------------------------------------
    # Storing model outputs
    # -------------------------------------
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

    # -------------------------------------
    # Run outputs
    # -------------------------------------
    parser.add_argument(
        "--results_path",
        type=str,
        default="logs/test.args",
        help="Where to save the result logs",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="defined either automatically by dispatcher.py or time in main.py. Keep without default",
    )

    # -------------------------------------
    # System
    # -------------------------------------
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Num workers for each data loader [default: 4]",
    )

    # cache
    parser.add_argument(
        "--cache_path", type=str, default=None, help="Dir to cache images."
    )

    # -------------------------------------
    # Logging
    # -------------------------------------

    parser.add_argument(
        "--logger_name",
        type=str,
        action=set_nox_type("logger"),
        choices=["tensorboard", "comet", "wandb"],
        default="tensorboard",
        help="experiment logger to use",
    )
    parser.add_argument(
        "--logger_tags", nargs="*", default=[], help="List of tags for logger"
    )
    parser.add_argument("--project_name", default="CancerCures", help="Comet project")
    parser.add_argument("--workspace", default=None, help="Comet workspace")
    parser.add_argument(
        "--log_gen_image",
        action="store_true",
        default=False,
        help="Whether to log sample generated image to comet",
    )
    parser.add_argument(
        "--log_profiler",
        action="store_true",
        default=False,
        help="Log profiler times to logger",
    )

    # -------------------------------------
    # Add object-level args
    # -------------------------------------

    def add_class_args(args_as_dict, parser):
        # for loop
        for argname, argval in args_as_dict.items():
            args_for_noxs = {
                a.dest: a for a in parser._actions if hasattr(a, "is_nox_action")
            }
            old_args = vars(parser.parse_known_args()[0])
            if argname in args_for_noxs:
                args_for_noxs[argname].add_args(parser, argval)
                newargs = vars(parser.parse_known_args()[0])
                newargs = {k: v for k, v in newargs.items() if k not in old_args}
                add_class_args(newargs, parser)

    parser.parse_known_args(namespace=global_namespace)
    add_class_args(vars(global_namespace), parser)

    return parser


def parse_args(args_strings=None):
    # run
    # Lightning 2.0 removes add_argparse_args
    parser = get_parser()
    sig = inspect.signature(Trainer.__init__)
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.default is inspect.Parameter.empty:
            parser.add_argument(f"--{name}")
        else:
            # Note that this does not store types, because many defaults are None
            parser.add_argument(f"--{name}", default=param.default)

    # legacy
    if not hasattr(Trainer, "add_argparse_args"):
        parser.add_argument(
            "--gpus",
            default=None,
            help="Number of GPUs to train on",
        )

    if args_strings is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args_strings)

    # legacy - fix arg changes
    if hasattr(args, "gpus") and not hasattr(Trainer, "add_argparse_args"):
        args.devices = args.gpus

    # using gpus
    if (isinstance(args.gpus, str) and len(args.gpus.split(",")) > 1) or (
        isinstance(args.gpus, int) and args.gpus > 1
    ):
        args.strategy = "ddp"
        # should not matter since we set our sampler, in 2.0 this is default True and used to be called replace_sampler_ddp
        args.use_distributed_sampler = False
    else:
        if not hasattr(Trainer, "add_argparse_args"):  # lightning 2.0
            # args.strategy = "auto"
            args.strategy = "ddp"  # should be overwritten later in main
        else:  # legacy
            args.strategy = None
            args.replace_sampler_ddp = False

    # username
    args.unix_username = pwd.getpwuid(os.getuid())[0]

    # learning initial state
    args.step_indx = 1

    # set args
    args_for_noxs = {a.dest: a for a in parser._actions if hasattr(a, "is_nox_action")}
    for argname, argval in vars(args).items():
        if argname in args_for_noxs:
            args_for_noxs[argname].set_args(args, argval)

    # parse augmentations
    args.train_rawinput_augmentations = parse_augmentations(
        args.train_rawinput_augmentation_names
    )
    args.train_tnsr_augmentations = parse_augmentations(
        args.train_tnsr_augmentation_names
    )
    args.test_rawinput_augmentations = parse_augmentations(
        args.test_rawinput_augmentation_names
    )
    args.test_tnsr_augmentations = parse_augmentations(
        args.test_tnsr_augmentation_names
    )

    # parse tune parameters
    # args = parse_tune_params(args)

    return args
