import torch
import torchvision
from vino.augmentations.abstract import Abstract_augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ToTensor(Abstract_augmentation):
    """
    torchvision.transforms.ToTensor wrapper.
    """

    def __init__(self):
        super(ToTensor, self).__init__()
        self.transform = ToTensorV2()
        self.name = "totensor"

    def __call__(self, input_dict, sample=None):
        input_dict["input"] = torch.from_numpy(input_dict["input"]).float()
        if input_dict.get("mask", None) is not None:
            input_dict["mask"] = torch.from_numpy(input_dict["mask"]).float()
        return input_dict


class Permute3d(Abstract_augmentation):
    """
    Permute tensor (T, C, H, W) ==> (C, T, H, W)
    """

    def __init__(self):
        super(Permute3d, self).__init__()

        def permute_3d(tensor):
            return tensor.permute(1, 0, 2, 3)

        self.transform = torchvision.transforms.Lambda(permute_3d)

    def __call__(self, input_dict, sample=None):
        input_dict["input"] = self.transform(input_dict["input"])
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
