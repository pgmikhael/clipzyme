# type: ignore

import sys

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

import rdkit

# data
import clipzyme.datasets.enzymemap
import clipzyme.datasets.enzyme_screening
import clipzyme.datasets.reaction

# lightning
import clipzyme.lightning.base

# optimizers
import clipzyme.learning.optimizers.basic

# scheduler
import clipzyme.learning.schedulers.basic
import clipzyme.learning.schedulers.warmup

# losses
import clipzyme.learning.losses.basic
import clipzyme.learning.losses.contrastive


# metrics
import clipzyme.learning.metrics.basic
import clipzyme.learning.metrics.representation

# callbacks
import clipzyme.callbacks.basic

# models
import clipzyme.models.classifier
import clipzyme.models.gat
import clipzyme.models.chemprop
import clipzyme.models.fair_esm
import clipzyme.models.egnn
import clipzyme.models.protmol
import clipzyme.models.wln
import clipzyme.models.seq2seq

# comet
import clipzyme.loggers.wandb
import clipzyme.loggers.tensorboard

from clipzyme.datasets.reaction import ReactionDataset
from clipzyme.lightning.clipzyme import CLIPZyme
from clipzyme.utils.registry import get_object
from clipzyme.utils.screening import collect_screening_results

__all__ = ["CLIPZyme", "ReactionDataset", "get_object", "collect_screening_results"]
