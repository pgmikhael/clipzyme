import torch
from torch import optim
from vino.utils.registry import register_object
from vino.utils.classes import Vino
from ray.tune.suggest import BasicVariantGenerator


@register_object("basic", "searcher")
class BasicSearch(BasicVariantGenerator, Vino):
    """Description

    See: https://docs.ray.io/en/releases-0.8.4/tune-searchalg.html#variant-generation-grid-search-random-search
    """

    def __init__(self, args):
        super().__init__()
