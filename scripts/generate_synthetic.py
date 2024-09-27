import logging
import shutil
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm

from lib.data.loader import load_intrinsics
from lib.data.synthetic import generate_synthetic_params
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

    for sequence_idx in tqdm(range(cfg.data.sequence_size)):
        # define the sequence directory
        sequence_dir = Path(cfg.data.data_dir) / f"s{sequence_idx:05}"
        # create default params for the sequence
        default_params = generate_synthetic_params(
            flame,
            window_size=cfg.data.params_settings.window_size,
            default=cfg.data.params_settings.default,
            sigmas=cfg.data.params_settings.sigmas,
            sparsity=cfg.data.params_settings.sparsity,
            select=cfg.data.params_filter,
        )
        for frame_idx in tqdm(range(cfg.data.frame_size)):
            # create the offset for the frame
            offset = generate_synthetic_params(
                flame,
                window_size=cfg.data.offset_settings.window_size,
                default=cfg.data.offset_settings.default,
                sigmas=cfg.data.offset_settings.sigmas,
                sparsity=cfg.data.offset_settings.sparsity,
                select=cfg.data.params_filter,
            )
            # add the offset to the default params for the sequence
            params: dict = {}
            for p_name, default in default_params.items():
                params[p_name] = default + offset[p_name]

            # save the params
            path = sequence_dir / f"params/{frame_idx:05}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({k: v.detach().cpu() for k, v in params.items()}, path)

            for scale in cfg.data.scales:
                # update the scale and render the params out
                renderer.update(scale=scale)
                out = flame.render(renderer=renderer, params=params)

                # save the data types
                for data_type in ["mask", "point", "normal", "color"]:
                    path = sequence_dir / f"cache/{scale}_{data_type}/{frame_idx:05}.pt"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    data = out[data_type].detach().cpu()[0]
                    torch.save(data, path)

                # save the color as image
                path = sequence_dir / f"color/{frame_idx:05}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                img = Image.fromarray(out["color"][0].detach().cpu().numpy())
                img.save(path)

            src = Path(cfg.data.intrinsics_dir) / "calibration.json"
            dst = sequence_dir / "calibration.json"
            shutil.copyfile(src, dst)


if __name__ == "__main__":
    optimize()
