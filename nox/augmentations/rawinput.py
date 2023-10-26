import albumentations as A
from nox.utils.registry import register_object
from nox.augmentations.abstract import Abstract_augmentation


@register_object("scale_2d", "augmentation")
class Scale_2d(Abstract_augmentation):
    """
    Given PIL image, enforce its some set size
    (can use for down sampling / keep full res)
    """

    def __init__(self, args, kwargs):
        super(Scale_2d, self).__init__()
        assert len(kwargs.keys()) == 0
        width, height = args.img_size
        self.set_cachable(width, height)
        self.transform = A.Resize(height, width)

    def __call__(self, input_dict, sample=None):
        out = self.transform(
            image=input_dict["input"], mask=input_dict.get("mask", None)
        )
        input_dict["input"] = out["image"]
        input_dict["mask"] = out["mask"]
        return input_dict


@register_object("rand_hor_flip", "augmentation")
class Random_Horizontal_Flip(Abstract_augmentation):
    """
    Randomly flips image horizontally
    """

    def __init__(self, args, kwargs):
        super(Random_Horizontal_Flip, self).__init__()
        self.args = args
        assert len(kwargs.keys()) == 0
        self.transform = A.HorizontalFlip()

    def __call__(self, input_dict, sample=None):
        if "seed" in sample:
            self.set_seed(sample["seed"])
        input_dict["input"] = self.transform(image=input_dict["input"])["image"]
        return input_dict


@register_object("rand_ver_flip", "augmentation")
class Random_Vertical_Flip(Abstract_augmentation):
    """
    Randomly flips image vertically
    """

    def __init__(self, args, kwargs):
        super(Random_Vertical_Flip, self).__init__()
        self.args = args
        assert len(kwargs.keys()) == 0
        self.transform = A.VerticalFlip()

    def __call__(self, input_dict, sample=None):
        if "seed" in sample:
            self.set_seed(sample["seed"])
        input_dict["input"] = self.transform(image=input_dict["input"])["image"]
        return input_dict


@register_object("random_resize_crop", "augmentation")
class RandomResizedCrop(Abstract_augmentation):
    """
    Randomly Resize and Crop: https://albumentations.ai/docs/api_reference/augmentations/crops/transforms/#albumentations.augmentations.crops.transforms.RandomResizedCrop
    kwargs:
        h: output height
        w: output width
        min_scale: min size of the origin size cropped
        max_scale: max size of the origin size cropped
        min_ratio: min aspect ratio of the origin aspect ratio cropped
        max_ratio: max aspect ratio of the origin aspect ratio cropped
    """

    def __init__(self, args, kwargs):
        super(RandomResizedCrop, self).__init__()
        self.args = args
        kwargs_len = len(kwargs.keys())
        assert (kwargs_len >= 2) and (kwargs_len <= 6)
        h, w = (int(kwargs["h"]), int(kwargs["w"]))
        min_scale = float(kwargs["min_scale"]) if "min_scale" in kwargs else 0.08
        max_scale = float(kwargs["max_scale"]) if "max_scale" in kwargs else 1.0
        min_ratio = float(kwargs["min_ratio"]) if "min_ratio" in kwargs else 0.75
        max_ratio = float(kwargs["max_ratio"]) if "max_ratio" in kwargs else 1.33
        self.transform = A.RandomResizedCrop(
            height=h,
            width=w,
            scale=(min_scale, max_scale),
            ratio=(min_ratio, max_ratio),
        )

    def __call__(self, input_dict, sample=None):
        if "seed" in sample:
            self.set_seed(sample["seed"])
        input_dict["input"] = self.transform(image=input_dict["input"])["image"]
        return input_dict


@register_object("random_crop", "augmentation")
class Random_Crop(Abstract_augmentation):
    """
    Randomly Crop: https://albumentations.ai/docs/api_reference/augmentations/crops/transforms/#albumentations.augmentations.crops.transforms.RandomCrop
    kwargs:
        h: output height
        w: output width
    """

    def __init__(self, args, kwargs):
        super(Random_Crop, self).__init__()
        self.args = args
        kwargs_len = len(kwargs.keys())
        assert kwargs_len in [2, 3]
        size = (int(kwargs["h"]), int(kwargs["w"]))
        self.transform = A.RandomCrop(*size)

    def __call__(self, input_dict, sample=None):
        if "seed" in sample:
            self.set_seed(sample["seed"])
        input_dict["input"] = self.transform(image=input_dict["input"])["image"]
        return input_dict


@register_object("blur", "augmentation")
class Blur(Abstract_augmentation):
    """
    Randomly blurs image with kernel size limit: https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Blur

    kwargs:
        limit: maximum kernel size for blurring the input image. Should be in range [3, inf)
    """

    def __init__(self, args, kwargs):
        super(Blur, self).__init__()
        limit = float(kwargs["limit"]) if "limit" in kwargs else 3
        self.transform = A.Blur(blur_limit=limit)

    def __call__(self, input_dict, sample=None):
        if "seed" in sample:
            self.set_seed(sample["seed"])
        input_dict["input"] = self.transform(image=input_dict["input"])["image"]
        return input_dict


@register_object("center_crop", "augmentation")
class Center_Crop(Abstract_augmentation):
    """
    Center crop: https://albumentations.ai/docs/api_reference/augmentations/crops/transforms/#albumentations.augmentations.crops.transforms.CenterCrop

    kwargs:
        h: height
        w: width
    """

    def __init__(self, args, kwargs):
        super(Center_Crop, self).__init__()
        self.args = args
        kwargs_len = len(kwargs.keys())
        assert kwargs_len in [2, 3]
        size = (int(kwargs["h"]), int(kwargs["w"]))
        self.set_cachable(*size)
        self.transform = A.CenterCrop(*size)

    def __call__(self, input_dict, sample=None):
        input_dict["input"] = self.transform(image=input_dict["input"])["image"]
        return input_dict


@register_object("color_jitter", "augmentation")
class Color_Jitter(Abstract_augmentation):
    """
    Center crop: https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.ColorJitter

    kwargs:
        brightness: default 0.2
        contrast: default 0.2
        saturation: default 0.2
        hue: default 0.2
    """

    def __init__(self, args, kwargs):
        super(Color_Jitter, self).__init__()
        self.args = args
        kwargs_len = len(kwargs.keys())
        assert kwargs_len <= 4
        b, c, s, h = (
            float(kwargs["brightness"]) if "brightness" in kwargs else 0.2,
            float(kwargs["contrast"]) if "contrast" in kwargs else 0.2,
            float(kwargs["saturation"]) if "saturation" in kwargs else 0.2,
            float(kwargs["hue"]) if "hue" in kwargs else 0.2,
        )
        self.transform = A.HueSaturationValue(
            brightness=b, contrast=c, saturation=s, hue=h
        )

    def __call__(self, input_dict, sample=None):
        if "seed" in sample:
            self.set_seed(sample["seed"])
        input_dict["input"] = self.transform(image=input_dict["input"])["image"]
        return input_dict


@register_object("random_hue_satur_val", "augmentation")
class Hue_Saturation_Value(Abstract_augmentation):
    """
        HueSaturationValue wrapper

    kwargs:
        val (val_shift_limit): default 0
        saturation (sat_shift_limit): default 0
        hue (hue_shift_limit): default 0
    """

    def __init__(self, args, kwargs):
        super(Hue_Saturation_Value, self).__init__()
        self.args = args
        kwargs_len = len(kwargs.keys())
        assert kwargs_len <= 3
        val, satur, hue = (
            int(kwargs["val"]) if "val" in kwargs else 0,
            int(kwargs["saturation"]) if "saturation" in kwargs else 0,
            int(kwargs["hue"]) if "hue" in kwargs else 0,
        )
        self.transform = A.HueSaturationValue(
            hue_shift_limit=hue, sat_shift_limit=satur, val_shift_limit=val, p=1
        )

    def __call__(self, input_dict, sample=None):
        if "seed" in sample:
            self.set_seed(sample["seed"])
        input_dict["input"] = self.transform(image=input_dict["input"])["image"]
        return input_dict


@register_object("random_brightness_contrast", "augmentation")
class Random_Brightness_Contrast(Abstract_augmentation):
    """
        RandomBrightnessContrast wrapper https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomBrightnessContrast

    kwargs:
        contrast (contrast_limit): default 0
        brightness (sat_shiftbrightness_limit_limit): default 0
    """

    def __init__(self, args, kwargs):
        super(Random_Brightness_Contrast, self).__init__()
        self.args = args
        kwargs_len = len(kwargs.keys())
        assert kwargs_len <= 2
        contrast = float(kwargs["contrast"]) if "contrast" in kwargs else 0
        brightness = float(kwargs["brightness"]) if "brightness" in kwargs else 0

        self.transform = A.RandomBrightnessContrast(
            brightness_limit=brightness, contrast_limit=contrast, p=1
        )

    def __call__(self, input_dict, sample=None):
        if "seed" in sample:
            self.set_seed(sample["seed"])
        input_dict["input"] = self.transform(image=input_dict["input"])["image"]
        return input_dict


@register_object("gaussian_blur", "augmentation")
class Gaussian_Blur(Abstract_augmentation):
    """
    wrapper for https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.GaussianBlur
    blur must odd and in range [3, inf). Default: (3, 7).

    kwargs:
        min_blur: default 3
        max_blur
    """

    def __init__(self, args, kwargs):
        super(Gaussian_Blur, self).__init__()
        self.args = args
        kwargs_len = len(kwargs.keys())
        assert kwargs_len >= 1
        min_blur = int(kwargs["min_blur"]) if "min_blur" in kwargs else 3
        max_blur = int(kwargs["max_blur"])
        assert (max_blur % 2 == 1) and (min_blur % 2 == 1)
        self.transform = A.GaussianBlur(blur_limit=(min_blur, max_blur), p=1)

    def __call__(self, input_dict, sample=None):
        if "seed" in sample:
            self.set_seed(sample["seed"])
        input_dict["input"] = self.transform(image=input_dict["input"])["image"]
        return input_dict


@register_object("rotate_range", "augmentation")
class Rotate_Range(Abstract_augmentation):
    """
    Rotate image counter clockwise by random degree https://albumentations.ai/docs/api_reference/augmentations/geometric/rotate/#albumentations.augmentations.geometric.rotate.Rotate

        kwargs
            deg: max degrees to rotate
    """

    def __init__(self, args, kwargs):
        super(Rotate_Range, self).__init__()
        assert len(kwargs.keys()) == 1
        self.max_angle = int(kwargs["deg"])
        self.transform = A.Rotate(limit=self.max_angle, p=0.5)

    def __call__(self, input_dict, sample=None):
        if "seed" in sample:
            self.set_seed(sample["seed"])
        out = self.transform(
            image=input_dict["input"], mask=input_dict.get("mask", None)
        )
        input_dict["input"] = out["image"]
        input_dict["mask"] = out["mask"]
        return input_dict


@register_object("grayscale", "augmentation")
class Grayscale(Abstract_augmentation):
    """
    Convert image to grayscale https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.ToGray
    """

    def __init__(self, args, kwargs):
        super(Grayscale, self).__init__()
        assert len(kwargs.keys()) == 0
        self.set_cachable(args.num_chan)

        self.transform = A.ToGray()

    def __call__(self, input_dict, sample=None):
        input_dict["input"] = self.transform(image=input_dict["input"])["image"]
        return input_dict
