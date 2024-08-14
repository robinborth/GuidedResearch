import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from lib.data.synthesis import generate_params
from lib.rasterizer import Rasterizer
from lib.renderer.camera import Camera
from lib.renderer.renderer import Renderer
from lib.utils.config import set_configs

log = logging.getLogger()


@hydra.main(version_base=None, config_path="../conf", config_name="generate_synthetic")
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
    flame = hydra.utils.instantiate(cfg.model).to(cfg.device)

    for idx in tqdm(range(cfg.data.dataset_size)):
        gt_params = generate_params(
            flame,
            window_size=cfg.data.params_settings.window_size,
            default=cfg.data.params_settings.default,
            sigmas=cfg.data.params_settings.sigmas,
            select=cfg.data.params_filter,
        )

        offset = generate_params(
            flame,
            window_size=cfg.data.offset_settings.window_size,
            default=cfg.data.offset_settings.default,
            sigmas=cfg.data.offset_settings.sigmas,
            select=cfg.data.params_filter,
        )
        params = {}
        for p_name in gt_params:
            params[p_name] = gt_params[p_name] + offset[p_name]

        # render the output
        m_out = flame(**gt_params)
        r_out = renderer.render_full(m_out["vertices"], faces=flame.faces)

        # params
        data = dict(
            default_params={},
            params={k: v.detach().cpu() for k, v in params.items()},
            gt_params={k: v.detach().cpu() for k, v in gt_params.items()},
            mask=r_out["mask"].detach().cpu(),
            point=r_out["point"].detach().cpu(),
            normal=r_out["normal"].detach().cpu(),
            color=r_out["color"].detach().cpu(),
            vertices=m_out["vertices"].detach().cpu(),
        )
        path = Path(cfg.data.data_dir) / f"{idx:05}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, path)


if __name__ == "__main__":
    optimize()
