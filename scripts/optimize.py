import warnings
from typing import List

import hydra
import lightning as L
import torch
import wandb
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger, WandbLogger
from omegaconf import DictConfig

from lib.utils.config import instantiate_callbacks, log_hyperparameters
from lib.utils.logger import create_logger, disable_pydevd_warning

log = create_logger("optimize")
# disable_pydevd_warning()
warnings.filterwarnings("ignore")


@hydra.main(version_base=None, config_path="../conf", config_name="optimize")
def optimize(cfg: DictConfig) -> None:
    log.info("==> loading config ...")
    L.seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("medium")  # using CUDA device RTX A400

    log.info(f"==> initializing datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"==> initializing model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("==> initializing callbacks ...")
    callbacks_schedulers = {**cfg.get("callbacks", {}), **cfg.get("scheduler", {})}
    callbacks: List[Callback] = instantiate_callbacks(callbacks_schedulers)

    log.info("==> initializing logger ...")
    warnings.filterwarnings("ignore")
    logger: Logger = hydra.utils.instantiate(cfg.logger)
    if isinstance(logger, WandbLogger):
        logger.watch(model, log="all")

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

    log.info("==> start optimizing ...")
    trainer.fit(model=model, datamodule=datamodule)

    log.info("==> finish wandb run ...")
    wandb.finish()


if __name__ == "__main__":
    optimize()
