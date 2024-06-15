import logging

import hydra
import lightning as L
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from lib.data.datamodule import DPHMDataModule
from lib.data.scheduler import CoarseToFineScheduler, FinetuneScheduler
from lib.model.flame import FLAME
from lib.model.logger import FlameLogger
from lib.model.loss import calculate_point2plane

log = logging.getLogger()


@hydra.main(version_base=None, config_path="../conf", config_name="optimize")
def optimize(cfg: DictConfig) -> None:
    log.info("==> loading config ...")
    L.seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("medium")  # using CUDA device RTX A400
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda"

    log.info(f"==> initializing datamodule <{cfg.data._target_}>")
    datamodule: DPHMDataModule = hydra.utils.instantiate(cfg.data, devie=device)
    datamodule.setup()  # this now contains the intrinsics, camera and rasterizer

    log.info(f"==> initializing model <{cfg.model._target_}>")
    model: FLAME = hydra.utils.instantiate(cfg.model)
    model.init_renderer(camera=datamodule.camera, rasterizer=datamodule.rasterizer)
    model = model.to(device)

    log.info("==> initializing scheduler ...")
    fts: FinetuneScheduler = hydra.utils.instantiate(cfg.scheduler.finetune)
    c2fs: CoarseToFineScheduler = hydra.utils.instantiate(cfg.scheduler.coarse2fine)

    log.info("==> initializing logger ...")
    logger: FlameLogger = hydra.utils.instantiate(cfg.logger)
    logger.init_logger(model=model, cfg=cfg)

    for iter_step in tqdm(range(cfg.trainer.max_iters)):  # outer loop
        # prepare iteration
        logger.iter_step = iter_step
        c2fs.schedule_dataset(datamodule=datamodule, iter_step=iter_step)
        optimizer = fts.configure_optimizers(model=model, iter_step=iter_step)
        optimizer.zero_grad()

        # fetch single batch
        dataloader = datamodule.train_dataloader()
        batch = next(iter(dataloader))

        # find correspondences
        with torch.no_grad():
            out = model.optimization_step(batch)
            mask = out["mask"]
            q = batch["point"][mask]  # (C, 3)
            n = out["normal"][mask]  # (C, 3)

        # optimization
        for optim_step in range(cfg.trainer.max_optims):
            logger.optim_step = optim_step
            logger.global_step = iter_step * cfg.trainer.max_optims + optim_step + 1
            optimizer.zero_grad()
            m_out = model.model_step(batch)
            p = model.renderer.mask_interpolate(
                vertices_idx=out["vertices_idx"],
                bary_coords=out["bary_coords"],
                attributes=m_out["vertices"],
                mask=mask,
            )  # (C, 3)
            point2plane = calculate_point2plane(q, p, n)  # (C,)
            logger.log("loss/point2plane", point2plane.mean())
            loss = point2plane.mean()
            loss.backward()
            logger.log_gradients(optimizer)
            optimizer.step()

        # debug and logging
        logger.log_metrics(batch=batch, model=out)
        if (iter_step % cfg.trainer.save_interval) == 0:
            logger.log_3d_points(batch=batch, model=out)
            logger.log_render(batch=batch, model=out)
            logger.log_input_batch(batch=batch, model=out)
            logger.log_loss(batch=batch, model=out)
            if model.init_mode == "flame":
                model.debug_params(batch=batch)


if __name__ == "__main__":
    optimize()
