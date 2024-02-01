from argparse import Namespace, FileType
import pickle
import collections.abc as container_abcs
import re
from tabnanny import check
from typing import Literal, Optional
from clipzyme.utils.registry import get_object
import torch
from torch.utils import data
from clipzyme.utils.sampler import DistributedWeightedSampler

try:
    from pytorch_lightning.utilities.cloud_io import load as pl_load
except:
    from lightning_fabric.utilities.cloud_io import _load as pl_load
from torch_geometric.data import Data, HeteroData, Batch
from rich import print as rprint

string_classes = (str, bytes)
int_classes = int
np_str_obj_array_pattern = re.compile(r"[SaUO]")

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, PyG Data or HeteroData, "
    "dicts, or lists; found {}"
)


def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, Batch):
        return batch
    if isinstance(elem, (Data, HeteroData)):
        # optional args for more complex graphs (see pytorch geometric https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html)
        follow_batch = None
        exclude_keys = None
        return Batch.from_data_list(batch, follow_batch, exclude_keys)
    elif isinstance(elem, torch.Tensor):
        # pad with zero
        if not all(v.shape == elem.shape for v in batch):
            max_len = max(v.shape[0] for v in batch)
            batch = [
                torch.concat([x, torch.zeros(max_len - x.shape[0], *x.shape[1:])])
                for x in batch
            ]

        elem = batch[0]
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])  # assumes all have the same size
            # numel = max([x.numel() for x in batch]) * len(batch)    # total size of tensors
            storage = elem.storage()._new_shared(
                numel
            )  # Creates a new storage in shared memory with the same data type
            out = elem.new(storage).view(-1, *list(elem.size()))

        return torch.stack(batch, 0, out=out)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if all(all(isinstance(t, tuple) for t in elem) for elem in it):
            return batch
        if not all(len(elem) == elem_size for elem in it):
            # if its a list of ints then append ints and return
            it = iter(batch)
            if all(all(isinstance(s, int_classes) for s in elem) for elem in it):
                return batch
            # if its a list of strings then append lists and return
            it = iter(batch)
            if all(
                all(isinstance(s, string_classes) for s in elem) for elem in it
            ) or all(
                all(
                    all(isinstance(s, string_classes) for s in t)
                    and isinstance(t, tuple)
                    for t in elem
                )
                for elem in it
            ):
                return batch
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def ignore_None_collate(batch):
    """
    default_collate wrapper that creates batches only of not None values.
    Useful for cases when the dataset.__getitem__ can return None because of some
    exception and then we will want to exclude that sample from the batch.
    """
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)


def get_train_dataset_loader(args: Namespace, split: Optional[str] = "train"):
    """Given arg configuration, return appropriate torch.DataLoader
    for train data loader

    Args:
        args (Namespace): args
        split (str, optional): dataset split. Defaults to "train".

    Returns:
        train_data_loader: iterator that returns batches
    """
    train_data = get_object(args.dataset_name, "dataset")(args, split)

    if args.class_bal:
        if args.strategy == "ddp":
            sampler = DistributedWeightedSampler(
                train_data,
                weights=train_data.weights,
                replacement=True,
                rank=args.global_rank,
                num_replicas=args.world_size,
                drop_last=True,
            )
        else:
            sampler = data.sampler.WeightedRandomSampler(
                weights=train_data.weights,
                num_samples=len(train_data),
                replacement=True,
            )
    else:
        if args.strategy == "ddp":
            sampler = torch.utils.data.distributed.DistributedSampler(
                train_data,
                shuffle=True,
                rank=args.global_rank,
                num_replicas=args.world_size,
                drop_last=True,
            )
        else:
            sampler = data.sampler.RandomSampler(train_data)

    train_data_loader = data.DataLoader(
        train_data,
        num_workers=args.num_workers,
        sampler=sampler,
        pin_memory=True,
        batch_size=args.batch_size,
        collate_fn=ignore_None_collate,
        drop_last=True,
    )

    return train_data_loader


def get_eval_dataset_loader(
    args: Namespace, split: Literal["train", "dev", "test"], shuffle=False
):
    """_summary_

    Args:
        args (Namespace): args
        split (Literal[&quot;train&quot;, &quot;dev&quot;, &quot;test&quot;]): dataset split.
        shuffle (bool, optional): whether to shuffle dataset. Defaults to False.

    Returns:
        data_loader: iterator that returns batches
    """
    if shuffle:
        rprint("[bold] Eval dataloader shuffle = True")

    eval_data = get_object(args.dataset_name, "dataset")(args, split)

    if args.strategy == "ddp":
        sampler = torch.utils.data.distributed.DistributedSampler(
            eval_data,
            shuffle=shuffle,
            rank=args.global_rank,
            num_replicas=args.world_size,
        )
    else:
        sampler = (
            torch.utils.data.sampler.RandomSampler(eval_data)
            if shuffle
            else torch.utils.data.sampler.SequentialSampler(eval_data)
        )
    data_loader = torch.utils.data.DataLoader(
        eval_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=ignore_None_collate,
        pin_memory=True,
        drop_last=False,
        sampler=sampler,
    )

    return data_loader


def is_dist_avail_and_initialized():
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [
            torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


def all_gather_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = torch.distributed.get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    # tensor_all = GatherLayer.apply(tensors)
    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """

    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


def get_lightning_model(args: Namespace):
    """Create new model or load from checkpoint

    Args:
        args (Namespace): global args

    Raises:
        FileType: checkpoint_path must be ".args" or ".ckpt" file

    Returns:
        model: pl.LightningModule instance
    """
    if args.from_checkpoint:
        if args.checkpoint_path.endswith(".args"):
            snargs = Namespace(**pickle.load(open(args.checkpoint_path, "rb")))
            # update saved args with new arguments
            for k, v in vars(args).items():
                if k not in snargs:
                    setattr(snargs, k, v)
            model = get_object(snargs.lightning_name, "lightning")(snargs)
            modelpath = snargs.model_path
        elif args.checkpoint_path.endswith(".ckpt"):
            model = get_object(args.lightning_name, "lightning")(args)
            modelpath = args.checkpoint_path
            checkpoint = pl_load(
                args.checkpoint_path, map_location=lambda storage, loc: storage
            )
            snargs = checkpoint["hyper_parameters"]["args"]
        else:
            raise FileType("checkpoint_path should be an args or ckpt file.")
        # update args with old args if not found
        for k, v in vars(snargs).items():
            if k not in args:
                setattr(args, k, v)
        model = model.load_from_checkpoint(
            checkpoint_path=modelpath,
            strict=not args.relax_checkpoint_matching,
            **{"args": args},
        )
        rprint(f"[bold] Loaded checkpoint from {modelpath}")
    else:
        model = get_object(args.lightning_name, "lightning")(args)
    return model
