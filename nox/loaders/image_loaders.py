from nox.loaders.abstract import AbstractLoader
from nox.utils.registry import register_object
import os
import cv2
import torch
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
import numpy as np

LOADING_ERROR = "LOADING ERROR! {}"


@register_object("cv_loader", "input_loader")
class OpenCVLoader(AbstractLoader):
    def configure_path(self, path, sample):
        return path

    def load_input(self, path, sample):
        """
        loads as grayscale image
        """
        return {"input": cv2.imread(path, 0)}

    @property
    def cached_extension(self):
        return ".png"


@register_object("dicom_loader", "input_loader")
class DicomLoader(AbstractLoader):
    def __init__(self, cache_path, augmentations, args):
        super(DicomLoader, self).__init__(cache_path, augmentations, args)
        self.window_center = -600
        self.window_width = 1500

    def configure_path(self, path, sample):
        return path

    def load_input(self, path, sample):
        try:
            dcm = pydicom.dcmread(path)
            dcm = apply_modality_lut(dcm.pixel_array, dcm)
            arr = apply_windowing(dcm, self.window_center, self.window_width)
        except Exception:
            raise Exception(LOADING_ERROR.format("COULD NOT LOAD DICOM."))
        return {"input": arr}

    @property
    def cached_extension(self):
        return ""


def transform_to_hu(self, dcm):
    """Transform DICOM pixel array to Hounsfield units

    Args:
        dcm (pydicom Dataset): dcm object read with pydicom

    Returns:
        np.array: numpy array of the DICOM image in Hounsfield
    """
    intercept = dcm.RescaleIntercept
    slope = dcm.RescaleSlope
    hu_image = dcm.pixel_array * slope + intercept

    return hu_image


def apply_windowing(image, center, width, bit_size=16):
    """Windowing function to transform image pixels for presentation.
    Must be run after a DICOM modality LUT is applied to the image.
    Windowing algorithm defined in DICOM standard:
    http://dicom.nema.org/medical/dicom/2020b/output/chtml/part03/sect_C.11.2.html#sect_C.11.2.1.2
    Reference implementation:
    https://github.com/pydicom/pydicom/blob/da556e33b/pydicom/pixel_data_handlers/util.py#L460
    Args:
        image (ndarray): Numpy image array
        center (float): Window center (or level)
        width (float): Window width
        bit_size (int): Max bit size of pixel
    Returns:
        ndarray: Numpy array of transformed images
    """
    y_min = 0
    y_max = 2**bit_size - 1
    y_range = y_max - y_min

    c = center - 0.5
    w = width - 1

    below = image <= (c - w / 2)  # pixels to be set as black
    above = image > (c + w / 2)  # pixels to be set as white
    between = np.logical_and(~below, ~above)

    image[below] = y_min
    image[above] = y_max
    if between.any():
        image[between] = ((image[between] - c) / w + 0.5) * y_range + y_min

    return image
