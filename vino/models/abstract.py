import torch.nn as nn
from vino.utils.classes import Vino
from abc import ABCMeta, abstractmethod

# from efficientnet_pytorch import EfficientNet
import math


class AbstractModel(nn.Module, Vino):
    __metaclass__ = ABCMeta

    def __init__(self):
        super(AbstractModel, self).__init__()
