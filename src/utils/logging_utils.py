"""Logging utility instantiator."""

import sys
from typing import Any, cast

from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def log_hyperparameters(object_dict: dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves number of model parameters.

    Args:
        object_dict: A dictionary containing the following objects: cfg, model, trainer.
    """
    hparams: dict = {}

    # Convert OmegaConf to dict and explicitly type it
    cfg = cast(dict[str, Any], OmegaConf.to_container(object_dict["cfg"]))
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    assert hparams is not None
    assert isinstance(hparams, dict)

    # Add model parameters
    hparams["model"] = cfg["model"]

    # Save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hparams["model/params/non_trainable"] = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    # Add other config sections
    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    # Use safer access methods for optional fields
    if "callbacks" in cfg:
        hparams["callbacks"] = cfg["callbacks"]
    if "extras" in cfg:
        hparams["extras"] = cfg["extras"]
    if "task_name" in cfg:
        hparams["task_name"] = cfg["task_name"]
    if "tags" in cfg:
        hparams["tags"] = cfg["tags"]
    if "ckpt_path" in cfg:
        hparams["ckpt_path"] = cfg["ckpt_path"]
    if "seed" in cfg:
        hparams["seed"] = cfg["seed"]

    hparams["execution_command"] = f"python {' '.join(sys.argv)}"

    # Send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)
