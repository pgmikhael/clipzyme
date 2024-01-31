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
import nox.datasets.enzymemap
import nox.datasets.enzyme_screening

# lightning
import nox.lightning.base

# optimizers
import nox.learning.optimizers.basic
import nox.learning.optimizers.lamb

# scheduler
import nox.learning.schedulers.basic
import nox.learning.schedulers.noam
import nox.learning.schedulers.warmup

# losses
import nox.learning.losses.basic
import nox.learning.losses.contrastive


# metrics
import nox.learning.metrics.basic
import nox.learning.metrics.representation

# callbacks
import nox.callbacks.basic
import nox.callbacks.swa

# models
import nox.models.classifier
import nox.models.gat
import nox.models.chemprop
import nox.models.fair_esm
import nox.models.egnn
import nox.models.protmol
import nox.models.wln

# comet
import nox.loggers.wandb
import nox.loggers.tensorboard
