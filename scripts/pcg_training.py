import logging
from typing import List

import hydra
import lightning as L
import wandb
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from lib.utils.config import instantiate_callbacks, log_hyperparameters, set_configs

log = logging.getLogger()


@hydra.main(version_base=None, config_path="../conf", config_name="pcg_training")
def train(cfg: DictConfig) -> None:
    log.info("==> loading config ...")
    cfg = set_configs(cfg)

    log.info("==> initializing logger ...")
    logger: Logger = hydra.utils.instantiate(cfg.logger)

    log.info(f"==> initializing datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"==> initializing model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("==> initializing callbacks ...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info(f"==> initializing trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }
    if logger:
        log.info("==> logging hyperparameters ...")
        log_hyperparameters(object_dict)

    log.info("==> start training ...")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    # closing wandb
    wandb.finish()


if __name__ == "__main__":
    train()
