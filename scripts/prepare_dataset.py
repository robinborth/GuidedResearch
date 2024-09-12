import logging
from pathlib import Path

import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from torchvision.io import decode_image
from torchvision.transforms import v2
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

from lib.data.dataset import DPHMDataset
from lib.data.loader import load_intrinsics
from lib.data.preprocessing import biliteral_filter, point2normal
from lib.model import Flame
from lib.model.flame.utils import load_static_landmark_embedding
from lib.renderer import Camera, Rasterizer, Renderer
from lib.tracker.logger import FlameLogger
from lib.utils.config import set_configs

log = logging.getLogger()


@hydra.main(version_base=None, config_path="../conf", config_name="optimize")
def optimize(cfg: DictConfig):
    log.info("==> loading config ...")
    depth_factor: float = cfg.data.depth_factor
    inf_depth: float = cfg.data.inf_depth
    scales: list[int] = cfg.data.scales
    data_dir = cfg.data.data_dir
    flame_dir = cfg.model.flame_dir
    cache_dir = Path(data_dir) / "cache"

    log.info("==> initializing camera and rasterizer ...")
    K = load_intrinsics(data_dir=data_dir, return_tensor="pt")
    camera = Camera(
        K=K,
        width=cfg.data.width,
        height=cfg.data.height,
        near=cfg.data.near,
        far=cfg.data.far,
        scale=1,
        device="cpu",
    )

    log.info("==> load mediapipe idx ...")
    flame_landmarks = load_static_landmark_embedding(flame_dir)
    media_idx = flame_landmarks["lm_mediapipe_idx"]

    max_length = len(list((Path(data_dir) / "depth").iterdir()))
    for idx in tqdm(range(max_length)):
        # load the color image
        path = Path(data_dir) / "color" / f"{idx:05}.png"
        color = pil_to_tensor(Image.open(path)).permute(1, 2, 0)

        # load the depth image and transform to m
        path = Path(data_dir) / "depth" / f"{idx:05}.png"
        img = Image.open(path)
        raw_depth = pil_to_tensor(img).to(torch.float32)[0]
        depth = raw_depth / depth_factor  # (H,W)

        # select the foreground based on a depth threshold
        f_mask = (depth < inf_depth) & (depth != 0)
        depth[~f_mask] = inf_depth

        # convert pointmap to normalmap
        point, _ = camera.depth_map_transform(depth)
        normal, n_mask = point2normal(point)

        # create the final mask based on normal and depth
        mask = f_mask & n_mask

        # mask the default values
        color[~mask] = 255
        normal[~mask] = 0

        # smooth the normal maps
        normal = biliteral_filter(
            image=normal,
            dilation=cfg.data.normal_dilation,
            sigma_color=150,
            sigma_space=150,
        )

        # create the point maps
        depth = biliteral_filter(
            image=depth,
            dilation=cfg.data.depth_dilation,
            sigma_color=150,
            sigma_space=150,
        )
        point, _ = camera.depth_map_transform(depth)
        point[~mask] = 0

        # create landmarks
        path = Path(data_dir) / "color/Mediapipe_landmarks" / f"{idx:05}.npy"
        landmark = torch.tensor(np.load(path))
        landmark[:, 0] *= camera.width
        landmark[:, 1] *= camera.height
        landmark = landmark[media_idx].long()
        u = landmark[:, 0]
        v = landmark[:, 1]

        landmarks = point[v, u]
        path = Path(data_dir) / f"landmark/{idx:05}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(landmarks, path)

        landmarks_mask = mask[v, u]
        path = Path(data_dir) / f"landmark_mask/{idx:05}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(landmarks_mask, path)

        for scale in scales:
            # downscale the images
            size = (int(camera.height / scale), int(camera.width / scale))
            image = v2.functional.resize(
                inpt=mask.to(torch.float32).unsqueeze(0),
                size=size,
            )
            down_mask = image[0] == 1.0

            down_color = v2.functional.resize(
                inpt=color.permute(2, 0, 1),
                size=size,
            ).permute(1, 2, 0)
            down_color[~down_mask] = 255

            down_normal = v2.functional.resize(
                inpt=normal.permute(2, 0, 1),
                size=size,
            ).permute(1, 2, 0)
            down_normal[~down_mask] = 0

            down_point = v2.functional.resize(
                inpt=point.permute(2, 0, 1),
                size=size,
            ).permute(1, 2, 0)
            down_point[~down_mask] = 0

            # save results
            path = cache_dir / f"{scale}_mask" / f"{idx:05}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(down_mask, path)

            path = cache_dir / f"{scale}_color" / f"{idx:05}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(down_color, path)

            path = cache_dir / f"{scale}_normal" / f"{idx:05}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(down_normal, path)

            path = cache_dir / f"{scale}_point" / f"{idx:05}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(down_point, path)


if __name__ == "__main__":
    optimize()
