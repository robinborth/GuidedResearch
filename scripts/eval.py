import logging
from pathlib import Path

import hydra
import lightning as L
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from lib.data.loader import load_intrinsics
from lib.model import Flame
from lib.renderer import Camera, Rasterizer, Renderer
from lib.tracker.logger import FlameLogger
from lib.utils.config import instantiate_callbacks, log_hyperparameters, set_configs

log = logging.getLogger()


@hydra.main(version_base=None, config_path="../conf", config_name="eval")
def main(cfg: DictConfig):
    try:
        optimize(cfg)
    except Exception as e:
        log.error(e)


def optimize(cfg: DictConfig):
    cfg = set_configs(cfg)
    log.info("==> loading config ...")

    log.info("==> initializing camera and rasterizer ...")
    K = load_intrinsics(data_dir=cfg.data.intrinsics_dir, return_tensor="pt")
    camera = Camera(
        K=K,
        width=cfg.data.width,
        height=cfg.data.height,
        near=cfg.data.near,
        far=cfg.data.far,
        scale=cfg.data.scale,
    )
    rasterizer = Rasterizer(width=camera.width, height=camera.height)
    renderer = Renderer(rasterizer=rasterizer, camera=camera)

    log.info(f"==> initializing model <{cfg.model._target_}> ...")
    flame: Flame = hydra.utils.instantiate(cfg.model).to(cfg.device)

    log.info(f"==> initializing logger <{cfg.logger._target_}> ...")
    logger: FlameLogger = hydra.utils.instantiate(cfg.logger)

    log.info(f"==> initializing datamodule <{cfg.data._target_}> ...")
    datamodule = hydra.utils.instantiate(cfg.data, renderer=renderer)

    log.info(f"==> initializing correspondence <{cfg.correspondence._target_}> ...")
    correspondence = hydra.utils.instantiate(cfg.correspondence)

    log.info(f"==> initializing weighting <{cfg.weighting._target_}> ...")
    weighting = hydra.utils.instantiate(cfg.weighting)

    log.info(f"==> initializing regularize <{cfg.regularize._target_}> ...")
    regularize = hydra.utils.instantiate(cfg.regularize)

    log.info(f"==> initializing residuals <{cfg.residuals._target_}> ...")
    residuals = hydra.utils.instantiate(cfg.residuals)

    log.info(f"==> initializing optimizer <{cfg.optimizer._target_}> ...")
    optimizer = hydra.utils.instantiate(cfg.optimizer)

    log.info(f"==> initializing framework <{cfg.framework._target_}> ...")
    model = hydra.utils.instantiate(
        cfg.framework,
        flame=flame,
        logger=logger,
        renderer=renderer,
        correspondence=correspondence,
        regularize=regularize,
        residuals=residuals,
        optimizer=optimizer,
        weighting=weighting,
    )

    if cfg.get("ckpt_path"):
        log.info("==> load model from checkpoint ...")
        checkpoint = torch.load(cfg.ckpt_path)
        model.load_state_dict(checkpoint["state_dict"])

    log.info("==> start evaluation ...")
    datamodule.setup("all")
    dataloader = datamodule.train_dataloader()
    for batch_idx, batch in tqdm(enumerate(dataloader)):
        batch = model.transfer_batch_to_device(batch, "cuda", 0)
        model.validation_step(batch, batch_idx)
    dataloader = datamodule.val_dataloader()
    for batch_idx, batch in tqdm(enumerate(dataloader)):
        batch = model.transfer_batch_to_device(batch, "cuda", 0)
        model.validation_step(batch, batch_idx)


if __name__ == "__main__":
    main()
