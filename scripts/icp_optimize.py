import hydra
import lightning as L
import torch
from lightning.pytorch.loggers import Logger, TensorBoardLogger, WandbLogger
from omegaconf import DictConfig
from tqdm import tqdm

from lib.data.datamodule import DPHMDataModule
from lib.data.scheduler import CoarseToFineScheduler, FinetuneScheduler
from lib.model.flame import FLAME
from lib.model.logger import FlameLogger
from lib.model.loss import calculate_point2plane
from lib.utils.logger import create_logger

log = create_logger("optimize")


@hydra.main(version_base=None, config_path="../conf", config_name="optimize")
def optimize(cfg: DictConfig) -> None:
    log.info("==> loading config ...")
    L.seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("medium")  # using CUDA device RTX A400
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info(f"==> initializing datamodule <{cfg.data._target_}>")
    datamodule: DPHMDataModule = hydra.utils.instantiate(cfg.data, devie=device)
    datamodule.setup()  # this now contains the intrinsics, camera and rasterizer

    log.info(f"==> initializing model <{cfg.model._target_}>")
    model: FLAME = hydra.utils.instantiate(cfg.model)
    model.init_renderer(camera=datamodule.camera, rasterizer=datamodule.rasterizer)
    model = model.to(device)

    log.info("==> initializing optimizer")
    optimizer_cfg = model.configure_optimizers()
    optimizer = optimizer_cfg["optimizer"]

    log.info("==> initializing scheduler ...")
    fts: FinetuneScheduler = hydra.utils.instantiate(cfg.scheduler.finetune)
    c2fs: CoarseToFineScheduler = hydra.utils.instantiate(cfg.scheduler.coarse2fine)

    log.info("==> initializing logger ...")
    _logger: Logger = hydra.utils.instantiate(cfg.logger)
    if isinstance(_logger, WandbLogger):
        _logger.watch(model)
    if isinstance(_logger, TensorBoardLogger):
        _logger.experiment
    logger = FlameLogger(logger=_logger, model=model)

    fts.freeze_before_training(model)
    for epoch in tqdm(range(cfg.trainer.max_epochs)):  # outer loop
        # settings
        logger.current_epoch = epoch
        c2fs.schedule_dataset(datamodule, epoch)
        fts.finetune_function(model, epoch, optimizer)

        # fetch single batch
        dataloader = datamodule.train_dataloader()
        batch = next(iter(dataloader))

        # find correspondences
        with torch.no_grad():
            out = model.optimization_step(batch)
            mask = out["mask"]
            q = batch["point"][mask]  # (C, 3)
            n = out["normal"][mask]  # (C, 3)

        # debug and logging
        logger.log_metrics(batch=batch, model=out)
        if (epoch % model.hparams["save_interval"]) == 0:
            # logger.debug_3d_points(batch=batch, model=out)
            logger.log_render(batch=batch, model=out)
            logger.log_input_batch(batch=batch, model=out)
            logger.log_loss(batch=batch, model=out)
            if model.hparams["init_mode"] == "flame":
                model.debug_params(batch=batch)

        for iter_step in range(100):
            optimizer.zero_grad()
            m_out = model.model_step(batch)
            p = model.renderer.mask_interpolate(
                vertices_idx=out["vertices_idx"],
                bary_coords=out["bary_coords"],
                attributes=m_out["vertices"],
                mask=mask,
            )  # (C, 3)
            point2plane = calculate_point2plane(q, p, n)  # (C,)
            loss = point2plane.mean()
            logger.log("train/loss", loss)
            loss.backward()
            optimizer.step()

    log.info("==> finished joint optimizing ...")


if __name__ == "__main__":
    optimize()
