import logging

import hydra
import lightning as L
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from lib.data.loader import load_intrinsics
from lib.model.flame import FLAME
from lib.rasterizer import Rasterizer
from lib.renderer.camera import Camera
from lib.trainer.logger import FlameLogger
from lib.trainer.trainer import Trainer

log = logging.getLogger()


@hydra.main(version_base=None, config_path="../conf", config_name="optimize")
def optimize(cfg: DictConfig) -> None:
    log.info("==> loading config ...")
    L.seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("medium")  # using CUDA device RTX A400
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda"

    log.info("==> initializing camera and rasterizer ...")
    K = load_intrinsics(data_dir=cfg.data.data_dir, return_tensor="pt")
    camera = Camera(
        K=K,
        width=cfg.data.width,
        height=cfg.data.height,
        near=cfg.data.near,
        far=cfg.data.far,
    )
    rasterizer = Rasterizer(
        width=camera.width,
        height=camera.height,
    )

    log.info(f"==> initializing model <{cfg.model._target_}>")
    model: FLAME = hydra.utils.instantiate(cfg.model).to(device)
    model.init_renderer(camera=camera, rasterizer=rasterizer)

    log.info("==> initializing logger ...")
    logger: FlameLogger = hydra.utils.instantiate(cfg.logger)
    logger.init_logger(model=model, cfg=cfg)

    log.info("==> initializing trainer ...")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        model=model,
        logger=logger,
    )

    log.info("==> optimize frames ...")
    for frame_idx in tqdm(range(cfg.data.max_frames)):
        if frame_idx != 0:
            model.init_frame(frame_idx)
            cfg.scheduler.optimizer.milestones = [0]
            cfg.scheduler.optimizer.params = [model.frame_p_names]
        optimizer_scheduler = hydra.utils.instantiate(cfg.scheduler.optimizer)
        coarse_to_fine_scheduler = hydra.utils.instantiate(
            cfg.scheduler.coarse2fine,
            camera=camera,
            rasterizer=rasterizer,
        )
        cfg.data.start_frame_idx = frame_idx
        datamodule = hydra.utils.instantiate(cfg.data, devie=device)
        trainer.optimize(
            datamodule=datamodule,
            optimizer_scheduler=optimizer_scheduler,
            coarse_to_fine_scheduler=coarse_to_fine_scheduler,
        )

    # final full screen image
    log.info("==> log final result ...")
    logger.iter_step = cfg.trainer.video_iter_step
    for frame_idx in tqdm(range(cfg.data.max_frames)):
        cfg.data.start_frame_idx = frame_idx
        datamodule = hydra.utils.instantiate(cfg.data, devie=device)
        coarse_to_fine_scheduler.full_screen(datamodule)
        logger.capture_screen(datamodule=datamodule, model=model)
    logger.log_video("render_normal", framerate=20)
    logger.log_video("render_merged", framerate=20)
    logger.log_video("error_point_to_plane", framerate=20)
    logger.log_video("batch_color", framerate=20)


if __name__ == "__main__":
    optimize()
