<div align="center">
<img src="/assets/purple-grape-vine.svg " width="256px"> 

**VINO**

a template for deep learning research projects


[![PyPI Status](https://badge.fury.io/py/vino.svg)](https://badge.fury.io/py/vino) [![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/PytorchLightning/pytorch-lightning/blob/master/LICENSE)
<!--
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/vino)](https://pypi.org/project/vino/)

[![PyPI Status](https://pepy.tech/badge/pytorch-lightning)](https://pepy.tech/project/pytorch-lightning) 

[![Conda](https://img.shields.io/conda/v/conda-forge/pytorch-lightning?label=conda&color=success)](https://anaconda.org/conda-forge/pytorch-lightning) 

[![DockerHub](https://img.shields.io/docker/pulls/pytorchlightning/pytorch_lightning.svg)](https://hub.docker.com/r/pytorchlightning/pytorch_lightning) 

[![codecov](https://codecov.io/gh/PyTorchLightning/pytorch-lightning/branch/master/graph/badge.svg)](https://codecov.io/gh/PyTorchLightning/pytorch-lightning) 


[![ReadTheDocs](https://readthedocs.org/projects/pytorch-lightning/badge/?version=stable)](https://pytorch-lightning.readthedocs.io/en/stable/) 


[![Slack](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://www.pytorchlightning.ai/community) 
-->
-----------------------------------

</div>

### Features

- [x] grid search
- [x] class-level args
    - to trigger arg set, arg must be as `[registry_object]_name_*`
- [x] callbacks
- [x] logging
    - [x] Comet ML
- [x] hyperparameter optimization with Ray
    - non-ddp hyperopt: use [TuneReportCallback](https://docs.ray.io/en/latest/ray-core/examples/using-ray-with-pytorch-lightning.html)
    - ddp + hyperopt: use [ray_lightning](https://github.com/ray-project/ray_lightning) + [tune.run](https://docs.ray.io/en/latest/tune/api_docs/execution.html?highlight=tune.run)
    - [tune options](https://docs.ray.io/en/latest/tune/api_docs/search_space.html#tune-sample-docs)
    - [x] train regular (old way)
    - [x] train ddp (old way)
    - [ ] tune regular
    - [ ] tune ddp 
- [x] standard formatting (autodocstring + black)
- [ ] streamlit
- [ ] tests
    - [ ] augmentations parser
    - [ ] mnist dataset
    - [ ] generic loaded model
    - [ ] training fit
    - [ ] training eval

Links

- https://docs.ray.io/en/latest/tune/api_docs/overview.html#tune-api-ref
- https://docs.ray.io/en/latest/tune/api_docs/suggestion.html#hyperopt-tune-suggest-hyperopt-hyperoptsearch
- https://docs.ray.io/en/latest/ray-core/examples/using-ray-with-pytorch-lightning.html#distributed-hyperparameter-optimization-with-ray-tune
