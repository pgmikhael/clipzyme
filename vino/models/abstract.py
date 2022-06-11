import torch.nn as nn
from nox.utils.classes import Nox
from abc import ABCMeta, abstractmethod

# from efficientnet_pytorch import EfficientNet
import math


class AbstractModel(nn.Module, Nox):
    __metaclass__ = ABCMeta

    def __init__(self):
        super(AbstractModel, self).__init__()
