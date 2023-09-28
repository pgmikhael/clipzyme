import torch
import torchvision
from nox.augmentations.abstract import Abstract_augmentation


class ToTensor(Abstract_augmentation):
    """
    torchvision.transforms.ToTensor wrapper.
    """

    def __init__(self):
        super(ToTensor, self).__init__()
        self.name = "totensor"

    def __call__(self, input_dict, sample=None):
        input_dict["input"] = torch.from_numpy(input_dict["input"]).float()
        if input_dict.get("mask", None) is not None:
            input_dict["mask"] = torch.from_numpy(input_dict["mask"]).float()
        return input_dict


class ComposeAug(Abstract_augmentation):
    """
    Composes multiple augmentations
    """

    def __init__(self, augmentations):
        super(ComposeAug, self).__init__()
        self.augmentations = augmentations

    def __call__(self, input_dict, sample=None):
        for transformer in self.augmentations:
            input_dict = transformer(input_dict, sample)

        return input_dict
