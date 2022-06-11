# dataset utils
import warnings
from typing import Literal, List
import numpy as np
import torchvision
from vino.datasets.abstract import AbstractDataset
from vino.utils.registry import register_object
from vino.utils.augmentations import get_augmentations_by_split
from vino.augmentations.basic import ComposeAug


@register_object("mnist", "dataset")
class MNIST_Dataset(AbstractDataset):
    """A pytorch Dataset for the MNIST data."""

    def __init__(self, args, split_group):
        """Initializes the dataset.

        Constructs a standard pytorch Dataset object which
        can be fed into a DataLoader for batching.

        Arguments:
            args(object): Config.
            augmentations(list): A list of augmentation objects.
            split_group(str): The split group ['train'|'dev'|'test'].

        Automatic downloading throws an error, install into args.data_dir manually with

            wget www.di.ens.fr/~lelarge/MNIST.tar.gz
            tar -zxvf MNIST.tar.gz
        """

        self.split_group = split_group
        self.args = args

        augmentations = get_augmentations_by_split(split_group, args)
        self.composed_all_augmentations = ComposeAug(augmentations)

        self.dataset = self.create_dataset(split_group)
        if len(self.dataset) == 0:
            return

        self.set_sample_weights(args)

        self.print_summary_statement(self.dataset, split_group)

    def create_dataset(
        self, split_group: Literal["train", "dev", "test"]
    ) -> List[dict]:

        if split_group == "train":
            dataset = torchvision.datasets.MNIST(
                self.args.cache_path, train=True, download=True
            )
        else:
            mnist_test = torchvision.datasets.MNIST(
                self.args.cache_path, train=False, download=True
            )

            if split_group == "dev":
                dataset = [mnist_test[i] for i in range(len(mnist_test) // 2)]
            elif split_group == "test":
                dataset = [
                    mnist_test[i] for i in range(len(mnist_test) // 2, len(mnist_test))
                ]
            else:
                raise Exception('Split group must be in ["train"|"dev"|"test"].')

        return dataset

    @staticmethod
    def set_args(args):
        args.num_classes = 10
        args.num_chan = 3
        args.img_size = (28, 28)
        args.img_mean = [0.0]
        args.img_std = [1.0]
        args.train_rawinput_augmentation_names = ["rand_hor_flip", "scale_2d"]
        args.test_rawinput_augmentation_names = ["scale_2d"]
        args.train_tnsr_augmentation_names = ["normalize_2d"]
        args.test_tnsr_augmentation_names = ["normalize_2d"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        x = np.array(x)
        sample = {"sample_id": "{}_{}_{}".format(self.split_group, index, y)}
        try:
            sample["x"] = self.composed_all_augmentations({"input": x}, sample)["input"]
            sample["x"] = sample["x"].repeat(self.args.num_chan, 1, 1).float()
            sample["y"] = y
            return sample

        except Exception:
            warnings.warn("Could not load sample")
