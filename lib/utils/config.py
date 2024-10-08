import logging
import warnings
from typing import List

import lightning as L
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from lightning import Callback
from omegaconf import DictConfig, OmegaConf

warnings.filterwarnings("ignore", message=".*num_workers.*")

log = logging.getLogger()


def instantiate_callbacks(callbacks_cfg: DictConfig | dict) -> List[Callback]:
    callbacks: List[Callback] = []

    if callbacks_cfg is None:
        log.info("No callbacks specified.")
        return callbacks

    for callback in callbacks_cfg.values():
        if "_target_" in callback.keys():
            callbacks.append(instantiate(callback))

    return callbacks


def load_config(config_name: str, overrides: list = []) -> DictConfig:
    """Loads the hydra config via code.

    Args:
        config_name (str): The name of the configuration.
        overrides (list): List of overrides to apply.

    Returns:
        DictConfig: The initialized config.
    """
    with initialize(config_path="../../conf", version_base=None):
        return compose(config_name=config_name, overrides=overrides)


def set_configs(cfg: DictConfig):
    L.seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("medium")  # using CUDA device RTX A400
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda"
    cfg.device = device
    return cfg


def log_hyperparameters(object_dict) -> None:
    hparams = {}

    cfg: dict = OmegaConf.to_container(object_dict["cfg"])  # type: ignore
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    trainer.logger.log_hyperparams(cfg)
