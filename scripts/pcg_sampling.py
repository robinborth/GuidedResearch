import logging

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from lib.data.loader import load_intrinsics
from lib.model.flame import FLAME
from lib.rasterizer import Rasterizer
from lib.renderer.camera import Camera
from lib.trainer.logger import FlameLogger
from lib.trainer.trainer import BaseTrainer
from lib.utils.config import set_configs

log = logging.getLogger()


@hydra.main(version_base=None, config_path="../conf", config_name="pcg_sampling")
def optimize(cfg: DictConfig) -> None:
    log.info("==> loading config ...")
    cfg = set_configs(cfg)

    log.info("==> initializing camera and rasterizer ...")
    K = load_intrinsics(data_dir=cfg.data.data_dir, return_tensor="pt")
    camera = Camera(
        K=K,
        width=cfg.data.width,
        height=cfg.data.height,
        near=cfg.data.near,
        far=cfg.data.far,
    )
    rasterizer = Rasterizer(width=camera.width, height=camera.height)

    log.info(f"==> initializing model <{cfg.model._target_}>")
    model: FLAME = hydra.utils.instantiate(cfg.model).to(cfg.device)
    model.init_renderer(camera=camera, rasterizer=rasterizer)

    log.info("==> initializing logger ...")
    logger: FlameLogger = hydra.utils.instantiate(cfg.logger)
    logger.init_logger(model=model, cfg=cfg)

    log.info("==> initializing datamodule ...")
    datamodule = hydra.utils.instantiate(cfg.data, devie=cfg.device)

    log.info("==> initilizing loss ...")
    loss = hydra.utils.instantiate(cfg.loss, _partial_=True)

    log.info("==> initilizing optimizer ...")
    optimizer = hydra.utils.instantiate(cfg.optimizer)

    assert cfg.get("joint_trainer")
    log.info("==> initializing trainer ...")
    trainer: BaseTrainer = hydra.utils.instantiate(
        cfg.pcg_sampling_trainer,
        model=model,
        loss=loss,
        optimizer=optimizer,
        logger=logger,
        datamodule=datamodule,
        camera=camera,
        rasterizer=rasterizer,
    )
    log.info("==> sample linear systems ...")
    trainer.optimize()


if __name__ == "__main__":
    optimize()
