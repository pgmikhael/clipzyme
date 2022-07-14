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
import nox.datasets.indrellinkpred
import nox.datasets.gsm_link

# augmentation
import nox.augmentations.rawinput
import nox.augmentations.tensor

# loader
import nox.loaders.image_loaders

# lightning
import nox.lightning.base
import nox.lightning.linker

# optimizers
import nox.learning.optimizers.basic

# scheduler
import nox.learning.schedulers.basic

# losses
import nox.learning.losses.basic
import nox.learning.losses.link_prediction

# metrics
import nox.learning.metrics.basic
import nox.learning.metrics.link_prediction

# callbacks
import nox.callbacks.basic
import nox.callbacks.swa

# models
import nox.models.vision
import nox.models.classifier
import nox.models.nbfnet
import nox.models.fair_esm
import nox.models.gat
import nox.models.linear

# comet
import nox.loggers.comet
import nox.loggers.tensorboard
import nox.loggers.wandb
import nox.loggers.tensorboard
