import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from lib.data.utils import (
    load_color,
    load_depth_masked,
    load_mediapipe_landmark_3d,
    load_points_3d,
)
from lib.renderer.camera import depth2camera, load_intrinsics


class DPHMDataset(Dataset):
    def __init__(
        self,
        # dataset settings
        data_dir: str = "/data",
        optimize_frames: int = 1,
        start_frame_idx: int = 0,
        # rasterizer settings
        image_scale: float = 1.0,
        image_width: int = 1920,
        image_height: int = 1080,
        **kwargs,
    ):
        self.optimize_frames = optimize_frames
        self.start_frame_idx = start_frame_idx
        self.image_size = int(image_height * image_scale), int(
            image_width * image_scale
        )
        self.image_scale = image_scale
        self.data_dir = data_dir
        self.K = load_intrinsics(
            data_dir=data_dir, return_tensor="pt", scale=image_scale
        )

    def iter_frame_idx(self):
        yield from range(
            self.start_frame_idx, self.start_frame_idx + self.optimize_frames
        )

    def load_lm3ds(self):
        self.lm3ds = []
        for frame_idx in self.iter_frame_idx():
            lm3d = load_mediapipe_landmark_3d(self.data_dir, idx=frame_idx)
            self.lm3ds.append(lm3d)

    def load_color(self):
        self.images = []
        for frame_idx in self.iter_frame_idx():
            image = load_color(
                data_dir=self.data_dir,
                idx=frame_idx,
                return_tensor="pt",
            )  # (H, W, 3)
            image = v2.functional.resize(
                inpt=image.permute(2, 0, 1),
                size=self.image_size,
            ).permute(1, 2, 0)
            self.images.append(image.to(torch.uint8))  # (H',W',3)

    def load_point_clouds(self):
        self.point_clouds = []
        for frame_idx in self.iter_frame_idx():
            point = load_points_3d(
                data_dir=self.data_dir,
                idx=frame_idx,
                return_tensor="np",
            )
            self.point_clouds.append(point)

    def load_point_images(self):
        self.point_images = []
        for frame_idx in self.iter_frame_idx():
            depth = load_depth_masked(
                data_dir=self.data_dir,
                idx=frame_idx,
                return_tensor="pt",
            )
            depth = depth2camera(depth=depth, K=self.K, scale=self.image_scale)
            self.point_images.append(depth)

    def __len__(self) -> int:
        return self.optimize_frames

    def __getitem__(self, idx: int):
        raise NotImplementedError()


class DPHMPointDataset(DPHMDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.load_color()
        self.load_point_images()

    def __getitem__(self, idx: int):
        point = self.point_images[idx]
        image = self.images[idx]  # (H', W', 3) this is scaled
        return {
            "shape_idx": 0,
            "frame_idx": idx,
            "point": point,
            "image": image,
        }
