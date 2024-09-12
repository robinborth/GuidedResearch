import logging

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from lib.data.loader import load_intrinsics
from lib.model import Flame
from lib.renderer import Camera, Rasterizer, Renderer
from lib.tracker.logger import FlameLogger
from lib.utils.config import set_configs

log = logging.getLogger()


@hydra.main(version_base=None, config_path="../conf", config_name="optimize")
def optimize(cfg: DictConfig):
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
    renderer = Renderer(camera=camera, rasterizer=rasterizer)

    log.info(f"==> initializing model <{cfg.model._target_}> ...")
    flame: Flame = hydra.utils.instantiate(cfg.model)

    log.info(f"==> initializing logger <{cfg.logger._target_}> ...")
    logger: FlameLogger = hydra.utils.instantiate(cfg.logger)

    log.info(f"==> initializing datamodule <{cfg.data._target_}> ...")
    datamodule = hydra.utils.instantiate(cfg.data)

    log.info(f"==> initializing correspondence <{cfg.correspondence._target_}> ...")
    correspondence = hydra.utils.instantiate(cfg.correspondence)

    log.info(f"==> initializing residuals <{cfg.residuals._target_}> ...")
    residuals = hydra.utils.instantiate(cfg.residuals)

    log.info(f"==> initializing optimizer <{cfg.optimizer._target_}> ...")
    optimizer = hydra.utils.instantiate(cfg.optimizer)

    log.info(f"==> initializing framework <{cfg.framework._target_}> ...")
    framework = hydra.utils.instantiate(
        cfg.framework,
        flame=flame,
        logger=logger,
        renderer=renderer,
        correspondence=correspondence,
        residuals=residuals,
        optimizer=optimizer,
    )

    log.info("==> initializing initial tracking ...")
    trainer = hydra.utils.instantiate(
        cfg.init_tracker,
        optimizer=framework,
        datamodule=datamodule,
    )
    log.info("==> start optimization ...")
    init_params = trainer.optimize()

    log.info("==> initializing joint tracking ...")
    trainer = hydra.utils.instantiate(
        cfg.joint_tracker,
        optimizer=framework,
        datamodule=datamodule,
        default_params=init_params,
    )
    log.info("==> start optimization ...")
    joint_params = trainer.optimize()

    log.info("==> initializing sequential tracking ...")
    framework.optimizer.store_system = True
    trainer = hydra.utils.instantiate(
        cfg.sequential_tracker,
        optimizer=framework,
        datamodule=datamodule,
        default_params=joint_params,
    )
    log.info("==> start optimization ...")
    sequential_params = trainer.optimize()

    log.info("==> prepare evaluation ...")
    for out in tqdm(sequential_params):
        logger.prepare_evaluation(
            renderer=renderer,
            datamodule=datamodule,
            flame=flame,
            params=out["params"],
            frame_idx=out["frame_idx"],
        )

    if trainer.final_video:
        log.info("==> create video ...")
        logger.log_tracking_video("render_normal", framerate=20)
        logger.log_tracking_video("render_merged", framerate=20)
        logger.log_tracking_video("error_point_to_plane", framerate=20)
        logger.log_tracking_video("batch_color", framerate=20)


if __name__ == "__main__":
    optimize()
