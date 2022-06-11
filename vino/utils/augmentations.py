from vino.augmentations.basic import ToTensor
from vino.utils.registry import get_object
from typing import Literal
from argparse import Namespace


def get_augmentations_by_split(
    split_group: Literal["train", "dev", "test"], args: Namespace
):
    """[summary]

    Parameters
    ----------
    ``split_group`` : str
        dataset split according to which the augmentation is selected (choices are ['train', 'dev', 'test'])
    ``args`` : Namespace
        global args

    Returns
    -------
    list
        sequence of initialized augmentations in list
    """
    if split_group in ["test", "dev"]:
        return get_augmentations(
            args.test_rawinput_augmentations,
            args.test_tnsr_augmentations,
            args,
        )
    else:
        return get_augmentations(
            args.train_rawinput_augmentations,
            args.train_tnsr_augmentations,
            args,
        )


def get_augmentations(image_augmentations, tensor_augmentations, args):
    """
    Args:
        image_augmentations: augmentations as list of tuples [(augmentation_name, dict(augmentation_kwargs) )]
        tensor_augmentations: augmentations as list of tuples [(augmentation_name, dict(augmentation_kwargs) )]
        args
    Returns:
        augmentations: list of augmentation instances
    """
    augmentations = []
    augmentations = _add_augmentations(augmentations, image_augmentations, args)
    augmentations.append(ToTensor())
    augmentations = _add_augmentations(augmentations, tensor_augmentations, args)
    return augmentations


def _add_augmentations(augmentations, new_augmentations, args):
    """
    Iterates through augmentations (list of tuples), parses name and kwargs, and returns list of augmentation objects for given registry
    """
    for trans in new_augmentations:
        name = trans[0]
        kwargs = trans[1]
        augmentations.append(get_object(name, "augmentation")(args, kwargs))

    return augmentations
