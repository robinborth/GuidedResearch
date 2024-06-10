import warnings
from dataclasses import dataclass
from typing import List

import hydra
import lightning as L
import torch
import wandb
from lightning import Callback, LightningModule
from lightning.pytorch.loggers import Logger, WandbLogger
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.data.datamodule import DPHMDataModule
from lib.data.scheduler import DummyScheduler
from lib.model.flame import FLAME
from lib.model.logger import FlameLogger
from lib.model.loss import calculate_point2plane
from lib.utils.config import instantiate_callbacks, log_hyperparameters
from lib.utils.logger import create_logger, disable_pydevd_warning

log = create_logger("optimize")
warnings.filterwarnings("ignore")


@dataclass
class Trainer:
    current_epoch: int
    optimizers: list[torch.optim.Optimizer]


@hydra.main(version_base=None, config_path="../conf", config_name="optimize")
def optimize(cfg: DictConfig) -> None:
    log.info("==> loading config ...")
    L.seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("medium")  # using CUDA device RTX A400

    log.info(f"==> initializing datamodule <{cfg.data._target_}>")
    datamodule: DPHMDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()  # this now contains the intrinsics, camera and rasterizer

    log.info(f"==> initializing model <{cfg.model._target_}>")
    model: FLAME = hydra.utils.instantiate(cfg.model)
    model.init_renderer(camera=datamodule.camera, rasterizer=datamodule.rasterizer)

    log.info("==> initializing optimizer")
    optimizer_cfg = model.configure_optimizers()
    optimizer = optimizer_cfg["optimizer"]

    # log.info("==> initializing callbacks ...")
    # finetune_scheduler = cfg.get("scheduler.finetune", DummyScheduler())
    # coarse2fine_scheduler = cfg.get("scheduler.coarse2fine", DummyScheduler())

    log.info("==> initializing callbacks ...")
    callbacks_schedulers = {**cfg.get("callbacks", {}), **cfg.get("scheduler", {})}
    callbacks: List[Callback] = instantiate_callbacks(callbacks_schedulers)

    log.info("==> initializing logger ...")
    _logger: Logger = hydra.utils.instantiate(cfg.logger)
    if isinstance(_logger, WandbLogger):
        _logger.watch(model)
    logger = FlameLogger(logger=_logger, model=model)

    log.info("==> initializing fabric ...")
    fabric = L.Fabric(
        accelerator=cfg.trainer.accelerator,
        callbacks=callbacks,
    )
    fabric.launch()

    log.info("==> setup model and optimizer ...")
    model, optimizer = fabric.setup(model, optimizer)

    # icp outer loop
    fabric.call("freeze_before_training", model)
    for epoch in tqdm(range(cfg.trainer.max_epochs)):
        logger.current_epoch = epoch
        fabric.call("schedule_dataset", datamodule, epoch)
        fabric.call("set_optimize_mode", model, epoch)
        fabric.call("finetune_function", model, epoch, optimizer)
        # ensures that it's on the gpu
        loader = datamodule.train_dataloader()
        dataloader: DataLoader = fabric.setup_dataloaders(loader)  # type: ignore
        for batch in dataloader:
            optimizer.zero_grad()
            output = model.optimization_step(batch)

            # calculate the loss
            point2plane = calculate_point2plane(
                q=output["point"],
                p=batch["point"].detach(),
                n=output["normal"].detach(),
            )  # (B, W, H)
            loss = point2plane[output["mask"]].mean()
            logger.log("train/loss", loss)

            # debug and logging
            logger.log_metrics(batch=batch, model=output)
            if (epoch % model.hparams["save_interval"]) == 0:
                # self.debug_3d_points(batch=batch, model=output)
                logger.log_render(batch=batch, model=output)
                logger.log_input_batch(batch=batch, model=output)
                logger.log_loss(batch=batch, model=output)
                if model.hparams["init_mode"] == "flame":
                    model.debug_params(batch=batch)

            fabric.backward(loss)
            # on_before_optimizer_step
            optimizer.step()
    log.info("==> finished joint optimizing ...")


if __name__ == "__main__":
    optimize()
