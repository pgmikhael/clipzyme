from nox.loaders.abstract import AbstractLoader
from nox.utils.registry import register_object
import os
import cv2
import torch
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
