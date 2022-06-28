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


# data
import nox.datasets.mnist

# augmentation
import nox.augmentations.rawinput
import nox.augmentations.tensor

# loader
import nox.loaders.image_loaders

# lightning
import nox.lightning.base

# optimizers
import nox.learning.optimizers.basic

# scheduler
import nox.learning.schedulers.basic

# losses
import nox.learning.losses.basic

# metrics
import nox.learning.metrics.basic

# callbacks
import nox.callbacks.basic
import nox.callbacks.swa

# models
import nox.models.vision
import nox.models.classifier

# comet
import nox.loggers.comet
import nox.loggers.wandb
