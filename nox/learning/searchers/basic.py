import torch
from torch import optim
from nox.utils.registry import register_object
from nox.utils.classes import Nox
from ray.tune.suggest import BasicVariantGenerator


@register_object("basic", "searcher")
class BasicSearch(BasicVariantGenerator, Nox):
    """Description

    See: https://docs.ray.io/en/releases-0.8.4/tune-searchalg.html#variant-generation-grid-search-random-search
    """

    def __init__(self, args):
        super().__init__()
