import logging

import hydra
from omegaconf import DictConfig

from lib.data.loader import load_intrinsics
from lib.rasterizer import Rasterizer
from lib.renderer.camera import Camera
from lib.renderer.renderer import Renderer
from lib.utils.config import set_configs

log = logging.getLogger()


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def optimize(cfg: DictConfig):
    log.info("==> loading config ...")
    cfg = set_configs(cfg)

    log.info("==> initializing camera and rasterizer ...")
    camera = Camera(
        width=cfg.data.width,
        height=cfg.data.height,
        near=cfg.data.near,
        far=cfg.data.far,
        scale=cfg.data.scale,
    )
    rasterizer = Rasterizer(width=camera.width, height=camera.height)
    renderer = Renderer(rasterizer=rasterizer, camera=camera)

    log.info(f"==> initializing model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model, renderer=renderer)

    log.info("==> initializing logger ...")
    logger = hydra.utils.instantiate(cfg.logger)

    log.info("==> initializing datamodule ...")
    datamodule = hydra.utils.instantiate(cfg.data)

    log.info("==> initializing trainer ...")
    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    log.info("==> start training ...")
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    optimize()
