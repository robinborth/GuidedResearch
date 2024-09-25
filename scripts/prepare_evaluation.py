import logging
import shutil
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm

from lib.data.dataset import DPHMDataset
from lib.data.loader import load_intrinsics
from lib.data.synthetic import generate_params
from lib.rasterizer import Rasterizer
from lib.renderer.camera import Camera
from lib.renderer.renderer import Renderer
from lib.utils.config import set_configs

log = logging.getLogger()


@hydra.main(version_base=None, config_path="../conf", config_name="optimize")
def optimize(cfg: DictConfig):
    log.info("==> loading config ...")
    cfg = set_configs(cfg)

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

    log.info(f"==> initializing model <{cfg.model._target_}>")
    flame = hydra.utils.instantiate(cfg.model).to(cfg.device)

    data_dirs = sorted(list(Path(cfg.data.data_dir).iterdir()))
    for data_dir in tqdm(data_dirs):
        dataset = DPHMDataset(data_dir=cfg.data.data_dir, scale=cfg.data.scale)
        for frame_idx in list(dataset.iter_frame_idx(dataset=data_dir.name)):
            params = dataset.load_param(dataset=data_dir.name, frame_idx=frame_idx)
            params = {k: v.to(cfg.device) for k, v in params.items()}
            out = flame.render(renderer=renderer, params=params, vertices_mask="face")

            path = data_dir / f"cache/{cfg.data.scale}_face_mask/{frame_idx:05}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(out["mask"][0].detach().cpu(), path)

            path = data_dir / f"vertices/{frame_idx:05}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(out["vertices"][0].detach().cpu(), path)


if __name__ == "__main__":
    optimize()
