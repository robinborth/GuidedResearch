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


@hydra.main(version_base=None, config_path="../conf", config_name="optimize")
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

    if "joint_trainer" in cfg:
        log.info("==> initializing joint trainer ...")
        trainer: BaseTrainer = hydra.utils.instantiate(
            cfg.joint_trainer,
            model=model,
            loss=loss,
            logger=logger,
            datamodule=datamodule,
            camera=camera,
            rasterizer=rasterizer,
        )
        log.info("==> joint optimization ...")
        trainer.optimize()

    if "sequential_trainer" in cfg:
        log.info("==> initializing sequential trainer ...")
        trainer = hydra.utils.instantiate(
            cfg.sequential_trainer,
            model=model,
            loss=loss,
            logger=logger,
            datamodule=datamodule,
            camera=camera,
            rasterizer=rasterizer,
        )
        log.info("==> sequential optimization ...")
        trainer.optimize()

    # final full screen image
    log.info("==> log final result ...")
    for frame_idx in tqdm(trainer.frames):
        logger.capture_screen(
            camera=camera,
            rasterizer=rasterizer,
            datamodule=datamodule,
            model=model,
            idx=frame_idx,
        )
    logger.log_video("render_normal", framerate=20)
    logger.log_video("render_merged", framerate=20)
    logger.log_video("error_point_to_plane", framerate=20)
    logger.log_video("batch_color", framerate=20)


if __name__ == "__main__":
    optimize()
