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
import vino.datasets.mnist

# augmentation
import vino.augmentations.rawinput
import vino.augmentations.tensor

# loader
import vino.loaders.image_loaders

# lightning
import vino.lightning.base

# optimizers
import vino.learning.optimizers.basic

# scheduler
import vino.learning.schedulers.basic

# losses
import vino.learning.losses.basic

# metrics
import vino.learning.metrics.basic

# callbacks
import vino.callbacks.basic
import vino.callbacks.swa

# models
import vino.models.vision
import vino.models.classifier

# comet
import vino.loggers.comet
