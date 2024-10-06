<div align="center">

# Machine Learning Project Template

[![python](https://img.shields.io/badge/-Python_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/) <br>
[![tests](https://github.com/ashleve/lightning-hydra-template/actions/workflows/test.yml/badge.svg)](https://github.com/ashleve/lightning-hydra-template/actions/workflows/test.yml)
[![code-quality](https://github.com/ashleve/lightning-hydra-template/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/ashleve/lightning-hydra-template/actions/workflows/code-quality-main.yaml)
[![codecov](https://codecov.io/gh/ashleve/lightning-hydra-template/branch/main/graph/badge.svg)](https://codecov.io/gh/ashleve/lightning-hydra-template) <br>
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/ashleve/lightning-hydra-template/pulls)
[![contributors](https://img.shields.io/github/contributors/ashleve/lightning-hydra-template.svg)](https://github.com/ashleve/lightning-hydra-template/graphs/contributors)

Click on [<kbd>Use this template</kbd>](https://github.com/rayanramoul/ml-project-template/generate) to start your own project!

<br>
A template for machine learning or deep learning projects.
</div>

<br>

## Features

- [x] Easy to implement your own model and dataloader
- [x] Configurable hyperparameters with Hydra
- [x] Logging with the solution that fits your needs
- [x] Works on CPU, multi-GPU, and multi-TPUs

## Installation

- [ ] Use this repository as a template
- [ ] Clone your repository
- [ ] Run `make install` to install the dependencies
- [ ] Add your model which inherits from `LightningModule` in `src/models`
- [ ] Add your dataset which inherits from `Datamodule` in `src/data`
- [ ] Add associated yaml configuration files in `configs/` folder following existing examples

## Tree Explained

```
.
├── commit-template.txt
├── configs
│   ├── callbacks
│   │   ├── default.yaml
│   │   ├── early_stopping.yaml
│   │   ├── model_checkpoint.yaml
│   │   ├── model_summary.yaml
│   │   ├── none.yaml
│   │   └── rich_progress_bar.yaml
│   ├── data
│   │   └── mnist.yaml
│   ├── debug
│   │   ├── default.yaml
│   │   ├── fdr.yaml
│   │   ├── limit.yaml
│   │   ├── overfit.yaml
│   │   └── profiler.yaml
│   ├── eval.yaml
│   ├── experiment
│   │   └── example.yaml
│   ├── extras
│   │   └── default.yaml
│   ├── hparams_search
│   │   └── mnist_optuna.yaml
│   ├── __init__.py
│   ├── local
│   ├── logger
│   │   ├── aim.yaml
│   │   ├── comet.yaml
│   │   ├── csv.yaml
│   │   ├── many_loggers.yaml
│   │   ├── mlflow.yaml
│   │   ├── neptune.yaml
│   │   ├── tensorboard.yaml
│   │   └── wandb.yaml
│   ├── model
│   │   └── mnist.yaml
│   ├── paths
│   │   └── default.yaml
│   ├── trainer
│   │   ├── cpu.yaml
│   │   ├── ddp_sim.yaml
│   │   ├── ddp.yaml
│   │   ├── default.yaml
│   │   ├── gpu.yaml
│   │   └── mps.yaml
│   └── train.yaml
├── data
├── Makefile
├── notebooks
├── pyproject.toml
├── README.md
├── ruff.toml
├── scripts
│   └── example_train.sh
├── src
│   ├── app.py
│   ├── data
│   │   ├── components
│   │   │   └── __init__.py
│   │   ├── __init__.py
│   │   └── mnist_datamodule.py
│   ├── Dockerfile
│   ├── eval.py
│   ├── __init__.py
│   ├── LICENSE
│   ├── models
│   │   ├── components
│   │   │   ├── __init__.py
│   │   │   └── simple_dense_net.py
│   │   ├── __init__.py
│   │   └── mnist_module.py
│   ├── README.md
│   ├── requirements.txt
│   ├── resources
│   │   └── screenshot.png
│   ├── train.py
│   └── utils
│       ├── __init__.py
│       ├── instantiators.py
│       ├── logging_utils.py
│       ├── pylogger.py
│       ├── rich_utils.py
│       └── utils.py
└── tests
    └── conftest.py

````

## Aknowledgements

This template was heavily inspired by great existing ones, like:

- [Lightning Hydra Template](https://github.com/ashleve/lightning-hydra-template/)
- [Pytorch Tempest](https://github.com/Erlemar/pytorch_tempest)
- [Yet Another Lightning Hydra Template](https://github.com/gorodnitskiy/yet-another-lightning-hydra-template)
<br>

But with a few opininated changes and improvements, go check them out!
