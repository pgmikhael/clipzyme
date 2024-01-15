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
import nox.datasets.mnist
import nox.datasets.benchmark_graphs
import nox.datasets.gsm
import nox.datasets.molecules
import nox.datasets.gsm_chemistry
import nox.datasets.reactions
import nox.datasets.brenda
import nox.datasets.dlkcat
import nox.datasets.enzymemap
import nox.datasets.ecreact
import nox.datasets.ecreact_graph
import nox.datasets.qm9
import nox.datasets.chembl
import nox.datasets.iocb_synthases
import nox.datasets.protein_graph
import nox.datasets.drugbank
import nox.datasets.tdc_adme
import nox.datasets.clean_ec
import nox.datasets.enzyme_screening

# lightning
import nox.lightning.base
import nox.lightning.linker
import nox.lightning.gsm_base
import nox.lightning.diffusion

# optimizers
import nox.learning.optimizers.basic
import nox.learning.optimizers.lamb

# scheduler
import nox.learning.schedulers.basic
import nox.learning.schedulers.noam
import nox.learning.schedulers.warmup

# losses
import nox.learning.losses.basic
import nox.learning.losses.link_prediction
import nox.learning.losses.contrastive
import nox.learning.losses.digress
import nox.learning.losses.attention
import nox.learning.losses.wln
import nox.learning.losses.protmol

# metrics
import nox.learning.metrics.basic
import nox.learning.metrics.link_prediction
import nox.learning.metrics.representation
import nox.learning.metrics.reactions
import nox.learning.metrics.digress
import nox.learning.metrics.wln
import nox.learning.metrics.protmol
import nox.learning.metrics.hierarchical_ec

# callbacks
import nox.callbacks.basic
import nox.callbacks.swa

# import nox.callbacks.ema

# models
import nox.models.vision
import nox.models.classifier
import nox.models.nbfnet
import nox.models.gat
import nox.models.chemprop
import nox.models.fair_esm
import nox.models.linear
import nox.models.metabonet
import nox.models.longformer
import nox.models.enzymenet
import nox.models.seq2seq
import nox.models.contrastive
import nox.models.egnn
import nox.models.graph_denoisers
import nox.models.digress
import nox.models.esm_decoder
import nox.models.protmol
import nox.models.noncanonnet
import nox.models.wln
import nox.models.reaction_center_classifier

# comet
import nox.loggers.comet
import nox.loggers.tensorboard
import nox.loggers.wandb
import nox.loggers.tensorboard
