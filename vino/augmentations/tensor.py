import torchvision
import torch
import numpy as np
import random
from nox.augmentations.abstract import Abstract_augmentation
from nox.utils.registry import register_object
import torchio as tio


@register_object("normalize_2d", "augmentation")
class Normalize_Tensor_2d(Abstract_augmentation):
    """
    Normalizes input by channel
    wrapper for torchvision.transforms.Normalize wrapper.
    """

    def __init__(self, args, kwargs):
        super(Normalize_Tensor_2d, self).__init__()
        assert len(kwargs) == 0
        channel_means = [args.img_mean] if len(args.img_mean) == 1 else args.img_mean
        channel_stds = [args.img_std] if len(args.img_std) == 1 else args.img_std

        self.transform = torchvision.transforms.Normalize(
            torch.Tensor(channel_means), torch.Tensor(channel_stds)
        )

        self.permute = args.img_file_type in [
            "png",
        ]

    def __call__(self, input_dict, sample=None):
        img = input_dict["input"]
        if len(img.size()) == 2:
            img = img.unsqueeze(0)

        if self.permute:
            img = img.permute(2, 0, 1)
            input_dict["input"] = self.transform(img).permute(1, 2, 0)
        else:
            input_dict["input"] = self.transform(img)

        return input_dict


@register_object("add_guassian_noise", "augmentation")
class GaussianNoise(Abstract_augmentation):
    """
    Add gaussian noise to img.

    kwargs:
        mean: mean of gaussian
        std: standard deviation of gaussian
    """

    def __init__(self, args, kwargs):
        super(GaussianNoise, self).__init__()
        kwargs_len = len(kwargs.keys())
        assert kwargs_len == 2
        self.mu = float(kwargs["mean"])
        self.sigma = float(kwargs["std"])

    def __call__(self, input_dict, sample=None):
        if "seed" in sample:
            self.set_seed(sample["seed"])
        input_dict["input"] = (
            input_dict["input"]
            + torch.randn(input_dict["input"].shape[-2:]) * self.sigma
            + self.mu
        )
        return input_dict

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mu, self.sigma
        )


@register_object("channel_shift", "augmentation")
class Channel_Shift_Tensor(Abstract_augmentation):
    """
    Randomly shifts values in a channel by a random number uniformly sampled
    from -shift:shift.

    kwargs:
        shift: float value for channel shift
    """

    def __init__(self, args, kwargs):
        super(Channel_Shift_Tensor, self).__init__()
        assert len(kwargs) == 1
        shift = float(kwargs["shift"])

        def apply_shift(img):
            shift_val = float(np.random.uniform(low=-shift, high=shift, size=1))
            return img + shift_val

        self.transform = torchvision.transforms.Lambda(apply_shift)

    def __call__(self, input_dict, sample=None):
        if "seed" in sample:
            self.set_seed(sample["seed"])
        input_dict["input"] = self.transform(input_dict["input"])
        return input_dict


@register_object("force_num_chan_2d", "augmentation")
class Force_Num_Chan_Tensor_2d(Abstract_augmentation):
    """
    Convert gray scale images to image with args.num_chan num channels.
    """

    def __init__(self, args, kwargs):
        super(Force_Num_Chan_Tensor_2d, self).__init__()
        assert len(kwargs) == 0
        self.args = args

    def __call__(self, input_dict, sample=None):
        img = input_dict["input"]
        mask = input_dict.get("mask", None)
        if mask is not None:
            input_dict["mask"] = mask.unsqueeze(0)

        num_dims = len(img.shape)
        if num_dims == 2:
            img = img.unsqueeze(0)
        existing_chan = img.size()[0]
        if not existing_chan == self.args.num_chan:
            input_dict["input"] = img.expand(self.args.num_chan, *img.size()[1:])

        return input_dict


@register_object("permute_channel_index", "augmentation")
class PermuteChannelIndex(Abstract_augmentation):
    """
    Permutes image so that channel is the first index: (H,W,C) ==> (C,H,W)
    """

    def __init__(self, args, kwargs):
        super(PermuteChannelIndex, self).__init__()
        assert len(kwargs) == 0
        self.set_cachable()

    def __call__(self, input_dict, sample=None):
        input_dict["input"] = input_dict["input"].permute(2, 0, 1).contiguous()
        return input_dict


@register_object("max_normalize_2d", "augmentation")
class MaxNormalize(Abstract_augmentation):
    """
    Divide each channel by maximum value so pixels are between [-1,1]
    """

    def __init__(self, args, kwargs):
        super(MaxNormalize, self).__init__()
        assert len(kwargs) == 0
        self.args = args

    def __call__(self, input_dict, sample=None):
        """
        input_dict['input']: tensor (C,H,W)
        """
        max_values, _ = input_dict["input"].view(self.args.num_chan, -1).max(-1)
        input_dict["input"] = input_dict["input"] / (
            torch.abs(max_values[:, None, None]) + 1e-9
        )
        return input_dict
