import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from lib.data.utils import load_color, load_mediapipe_landmark_3d, load_points_3d


class DPHMPointCloudDataset(Dataset):
    def __init__(
        self,
        # dataset settings
        data_dir: str = "/data",
        chunk_size: int = 50000,
        num_frames: int = 1,
        start_frame_idx: int = 0,
        # rasterizer settings
        scale_factor: int = 1,
        image_width: int = 1920,
        image_height: int = 1080,
    ):
        self.num_frames = num_frames
        self.chunk_size = chunk_size

        # load the frames in the memory
        points = []
        for frame_idx in range(start_frame_idx, start_frame_idx + num_frames):
            point_cloud = load_points_3d(
                data_dir=data_dir,
                idx=frame_idx,
                return_tensor="np",
            )
            points.append(point_cloud)
        self.points = points

        # for debugging
        image_size = (image_height // scale_factor, image_width // scale_factor)
        images = []
        for frame_idx in range(start_frame_idx, start_frame_idx + num_frames):
            image = load_color(
                data_dir=data_dir,
                idx=frame_idx,
                return_tensor="pt",
            )  # (H, W, 3)
            image = v2.functional.resize(
                inpt=image.permute(2, 0, 1),
                size=image_size,
            ).permute(1, 2, 0)
            images.append(image.to(torch.uint8))  # (H',W',3)
        self.images = images

        lm3ds = []
        for frame_idx in range(start_frame_idx, start_frame_idx + num_frames):
            lm3d = load_mediapipe_landmark_3d(data_dir, idx=frame_idx)
            lm3ds.append(lm3d)
        self.lm3ds = lm3ds

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, idx: int):
        point_cloud = self.points[idx]
        chunk_idx = np.random.choice(len(point_cloud), self.chunk_size, replace=False)
        image = self.images[idx]  # (H', W', 3) this is scaled
        lm3d = self.lm3ds[idx]
        return {
            "points": point_cloud[chunk_idx],
            "frame_idx": idx,
            "image": image,
            "mediapipe_lm3d": lm3d,
        }
