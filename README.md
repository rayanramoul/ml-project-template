<div align="center">

<img width="200" src="https://github.com/rayanramoul/ml-project-template/blob/master/assets/img/icon.png?raw=true">
</img>
<h1>Machine Learning Project Template</h1>

[![python](https://img.shields.io/badge/-Python_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/) <br>

Click on [<kbd>Use this template</kbd>](https://github.com/rayanramoul/ml-project-template/generate) to start your own project! or go to the [Documentation](https://rayanramoul.github.io/ml-project-template/) for more information.

<br>
A template for machine learning or deep learning projects.
<br>
<img src="https://raw.githubusercontent.com/catppuccin/catppuccin/main/assets/palette/macchiato.png" width="400" />
</div>

<br>

## 🧠 Features

- [x] Easy to implement your own model and dataloader
- [x] Configurable hyperparameters with Hydra
- [x] Logging with the solution that fits your needs
- [x] Works on CPU, multi-GPU, and multi-TPUs

## ⚙️ Installation

- [ ] Use this repository as a template
- [ ] Clone your repository
- [ ] Run `make install` to install the dependencies
- [ ] Add your model which inherits from `LightningModule` in `src/models`
- [ ] Add your dataset which inherits from `Datamodule` in `src/data`
- [ ] Add associated yaml configuration files in `configs/` folder following existing examples

## 🌳 Tree Explained

```
.
├── commit-template.txt
├── configs
│   ├── callbacks
│   ├── data
│   ├── debug
│   ├── eval.yaml
│   ├── experiment
│   ├── extras
│   ├── hparams_search
│   ├── local
│   ├── logger
│   ├── model
│   ├── paths
│   ├── trainer
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
│   │   └── mnist_datamodule.py
│   ├── Dockerfile
│   ├── eval.py
│   ├── models
│   │   ├── components
│   │   │   ├── __init__.py
│   │   │   └── simple_dense_net.py
│   │   ├── __init__.py
│   │   └── mnist_module.py
│   ├── train.py
│   └── utils
│       ├── instantiators.py
│       ├── logging_utils.py
│       ├── pylogger.py
│       ├── rich_utils.py
│       └── utils.py
└── tests
    └── conftest.py

````

## 🔮 Incoming features for this template

- [x] Add support for multi-GPU training
- [x] UV package manager setup
- [x] pre-commits hooks
- [x] Hydra instantiation of models and dataloaders
- [x] Add torch.compile of models
- [x] Integrate TorchTyping
- [x] Dockerize the project (Dockerfile, run tests and training through docker, optionally docker-compose)
- [x] Add example of efficient multi-processing using pool map
- [x] Add example using polars
- [x] Example mock tests
- [x] Util scripts to download dataset from kaggle for example
- [x] Cloud oriented scripts (launch your training on AWS, GCP, Azure)
- [x] Add config architecture and example of LitServe

## 🤝 Contributing

For more information on how to contribute to this project, please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file.

## 🌟 Aknowledgements

This template was heavily inspired by great existing ones, like:

- [Lightning Hydra Template](https://github.com/ashleve/lightning-hydra-template/)
- [Pytorch Tempest](https://github.com/Erlemar/pytorch_tempest)
- [Yet Another Lightning Hydra Template](https://github.com/gorodnitskiy/yet-another-lightning-hydra-template)
- [Pytorch Style Guide](https://github.com/IgorSusmelj/pytorch-styleguide)
<br>

But with a few opininated changes and improvements, go check them out!
