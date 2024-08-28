import logging

import hydra
from lightning import Trainer
from omegaconf import DictConfig

from lib.model import Flame
from lib.renderer import Camera, Rasterizer, Renderer
from lib.tracker.logger import FlameLogger
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

    log.info(f"==> initializing model <{cfg.model._target_}> ...")
    flame: Flame = hydra.utils.instantiate(cfg.model)

    log.info(f"==> initializing logger <{cfg.logger._target_}> ...")
    logger: FlameLogger = hydra.utils.instantiate(cfg.logger)

    log.info(f"==> initializing datamodule <{cfg.data._target_}> ...")
    datamodule = hydra.utils.instantiate(cfg.data, renderer=renderer)

    log.info(f"==> initializing correspondence <{cfg.correspondence._target_}> ...")
    correspondence = hydra.utils.instantiate(cfg.correspondence)

    log.info(f"==> initializing weighting <{cfg.weighting._target_}> ...")
    weighting = hydra.utils.instantiate(cfg.weighting)

    log.info(f"==> initializing residuals <{cfg.residuals._target_}> ...")
    residuals = hydra.utils.instantiate(cfg.residuals)

    log.info(f"==> initializing optimizer <{cfg.optimizer._target_}> ...")
    optimizer = hydra.utils.instantiate(cfg.optimizer)

    log.info(f"==> initializing framework <{cfg.framework}> ...")
    model = hydra.utils.instantiate(
        cfg.framework,
        flame=flame,
        logger=logger,
        renderer=renderer,
        correspondence=correspondence,
        residuals=residuals,
        optimizer=optimizer,
        weighting=weighting,
    )

    log.info("==> initializing trainer ...")
    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    if cfg.train:
        log.info("==> start training ...")
        trainer.fit(model=model, datamodule=datamodule)

    if cfg.eval:  # custom because of fwAD
        log.info("==> start evaluation ...")
        model = model.to("cuda")
        datamodule.setup("val")
        dataloader = datamodule.val_dataloader()
        for batch_idx, batch in enumerate(dataloader):
            batch = model.transfer_batch_to_device(batch, "cuda", 0)
            model.validation_step(batch, batch_idx)


if __name__ == "__main__":
    optimize()
