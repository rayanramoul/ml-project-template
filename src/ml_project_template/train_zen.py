"""Main training script using hydra-zen."""

from collections.abc import Callable
from typing import Any

import pytorch_lightning as pl
import torch
from hydra_zen import builds, make_config, make_custom_builds_fn, zen
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import Logger

from ml_project_template.utils import (
    RankedLogger,
    log_hyperparameters,
)

log = RankedLogger(__name__, rank_zero_only=True)

# Create a partial builds function that preserves signatures and supports partial application
pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)

# Pre-seed function to ensure reproducibility
pre_seed = zen(lambda seed: pl.seed_everything(seed, workers=True))


def train_impl(
    cfg: DictConfig,
    datamodule: LightningDataModule,
    model: LightningModule,
    callbacks: list[Callback],
    logger: list[Logger],
    trainer: Trainer,
    train: bool = True,
    test: bool = True,
    ckpt_path: str | None = None,
    model_compile: bool = False,
    optimized_metric: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Implementation of the training function."""
    if model_compile:
        log.info("Compiling model...")
        torch.compile(model)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    train_metrics = {}
    if train:
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        train_metrics = trainer.callback_metrics

    test_metrics = {}
    if test:
        log.info("Starting testing!")
        if train and hasattr(trainer, "checkpoint_callback") and trainer.checkpoint_callback is not None:
            best_ckpt_path = trainer.checkpoint_callback.best_model_path  # type: ignore
            if best_ckpt_path == "":
                log.warning("Best ckpt not found! Using current weights for testing...")
                best_ckpt_path = None
            else:
                log.info(f"Best ckpt path: {best_ckpt_path}")
                ckpt_path = best_ckpt_path

        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        test_metrics = trainer.callback_metrics

    # Merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


# Wrap the training function with zen to make it compatible with Hydra

# Use pre_call to ensure seeding happens before any instantiation
train_task = zen(train_impl, pre_call=pre_seed)


def create_default_config(
    model_cls: type[LightningModule] = None,
    datamodule_cls: type[LightningDataModule] = None,
    callbacks_fn: Callable = None,
    loggers_fn: Callable = None,
):
    """Create default configuration with placeholders for custom classes."""
    # Set default values appropriately
    model_target = model_cls.__module__ + "." + model_cls.__name__ if model_cls else "???"
    data_target = datamodule_cls.__module__ + "." + datamodule_cls.__name__ if datamodule_cls else "???"

    return make_config(
        # Core components
        seed=42,
        model=builds(model_target) if model_cls else None,
        data=builds(data_target) if datamodule_cls else None,
        trainer=builds(pl.Trainer, max_epochs=100, accelerator="auto"),
        # Optional components
        callbacks=callbacks_fn() if callbacks_fn else None,
        logger=loggers_fn() if loggers_fn else None,
        # Training options
        train=True,
        test=True,
        ckpt_path=None,
        model_compile=False,
        optimized_metric=None,
    )


# Register and expose task for command-line usage
if __name__ == "__main__":
    from hydra_zen import ZenStore, zen

    # Create a store to register configurations
    store = ZenStore(deferred_hydra_store=False)

    # Register the default configuration (as a placeholder)

    # Replace with your actual model and datamodule classes
    store(create_default_config(), name="train_config")

    # Launch the task with hydra_main
    train_task.hydra_main(
        config_name="train_config",
        version_base="1.3",
        config_path="./configs",
    )
