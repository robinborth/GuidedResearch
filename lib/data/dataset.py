import numpy as np
from torch.utils.data import Dataset

from lib.data.utils import load_points_3d


class DPHMPointCloudDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "/data",
        chunk_size: int = 50000,
        num_frames: int = 1,
        start_frame_idx: int = 0,
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

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, idx: int):
        point_cloud = self.points[idx]
        chunk_idx = np.random.choice(len(point_cloud), self.chunk_size, replace=False)
        return {"points": point_cloud[chunk_idx]}
