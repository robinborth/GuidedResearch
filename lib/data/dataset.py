import random
from pathlib import Path

import torch
from torch.utils.data import Dataset


class DPHMDataset(Dataset):
    def __init__(
        self,
        scale: int = 1,
        data_dir: str = "/data",
        **kwargs,
    ):
        self.sequence_length = len(list((Path(data_dir) / "depth").iterdir()))
        self.scale = scale
        self.data_dir = data_dir

    def iter_frame_idx(self):
        yield from range(self.sequence_length)

    def cache_path(self, name: str, frame_idx: int) -> Path:
        return (
            Path(self.data_dir)
            / "cache"
            / f"{self.scale}_{name}"
            / f"{frame_idx:05}.pt"
        )

    def load_cached(self, name: str, frame_idx: int):
        path = self.cache_path(name, frame_idx)
        return torch.load(path)

    def load(self, name: str):
        data = []
        for frame_idx in self.iter_frame_idx():
            landmarks = self.load_cached(name, frame_idx)
            data.append(landmarks)
        return data

    def load_landmark(self):
        landmark = []
        for frame_idx in self.iter_frame_idx():
            path = Path(self.data_dir) / f"landmark/{frame_idx:05}.pt"
            landmark.append(torch.load(path))
        return landmark

    def load_landmark_mask(self):
        landmark = []
        for frame_idx in self.iter_frame_idx():
            path = Path(self.data_dir) / f"landmark_mask/{frame_idx:05}.pt"
            landmark.append(torch.load(path))
        return landmark

    def load_params(self):
        params = []
        for frame_idx in self.iter_frame_idx():
            path = Path(self.data_dir) / f"params/{frame_idx:05}.pt"
            _params = torch.load(path)
            _params = {k: v[0] for k, v in _params.items()}
            params.append(_params)
        return params

    def __len__(self) -> int:
        return self.sequence_length

    def __getitem__(self, idx: int):
        raise NotImplementedError()


class DPHMPointDataset(DPHMDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mask = self.load("mask")
        self.normal = self.load("normal")
        self.color = self.load("color")
        self.point = self.load("point")
        self.landmark = self.load_landmark()
        self.landmark_mask = self.load_landmark_mask()
        self.frame_idxs = list(self.iter_frame_idx())

    def __getitem__(self, idx: int):
        # (H', W', 3) this is scaled
        mask = self.mask[idx]
        point = self.point[idx]
        normal = self.normal[idx]
        color = self.color[idx]
        landmark = self.landmark[idx]
        landmark_mask = self.landmark_mask[idx]
        frame_idx = self.frame_idxs[idx]
        return {
            "frame_idx": frame_idx,
            "mask": mask,
            "point": point,
            "normal": normal,
            "color": color,
            "landmark": landmark,
            "landmark_mask": landmark_mask,
        }


class DPHMParamsDataset(DPHMDataset):
    def __init__(
        self,
        start_frame: int = 1,
        mode: str = "fix",
        end_frame: int = 2,
        jump_size: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mask = self.load("mask")
        self.normal = self.load("normal")
        self.color = self.load("color")
        self.point = self.load("point")
        self.params = self.load_params()
        self.frame_idxs = list(self.iter_frame_idx())
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.jump_size = jump_size
        assert mode in ["fix", "range"]
        self.mode = mode
        assert self.start_frame >= self.jump_size

    def init_idx(self, idx: int):
        frame_idx = idx + self.start_frame
        if self.mode == "fix":
            return frame_idx - self.jump_size
        s_idx = frame_idx - self.jump_size
        e_idx = frame_idx + self.jump_size
        return random.randint(s_idx, e_idx)

    def __len__(self):
        return self.end_frame - self.start_frame

    def __getitem__(self, idx: int):
        # (H', W', 3) this is scaled
        frame_idx = idx + self.start_frame
        mask = self.mask[frame_idx]
        point = self.point[frame_idx]
        normal = self.normal[frame_idx]
        color = self.color[frame_idx]
        gt_params = self.params[frame_idx]
        init_idx = self.init_idx(idx)
        params = self.params[init_idx]
        init_color = self.color[init_idx]
        return {
            "frame_idx": frame_idx,
            "mask": mask,
            "point": point,
            "normal": normal,
            "color": color,
            "params": gt_params,
            "init_params": params,
            "init_color": init_color,
        }


class SplitDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "/flame",
        split: str = "train",
        samples: list[float] = [0.8, 0.1, 0.1],
    ):
        self.data_dir = data_dir
        self.split = split

        paths = sorted(list(Path(self.data_dir).iterdir()))
        i, j = self.split_dataset(split, samples, len(paths))

        data = []
        for i, path in enumerate(paths[i:j]):
            out = torch.load(path)
            out["sample_id"] = path.stem
            data.append(out)
        self.data = data

    def split_dataset(self, split: str, splits: list[float], num_samples: int):
        if split == "train":
            i = 0
            j = int(num_samples * splits[0])
        elif split == "val":
            i = int(num_samples * splits[0])
            j = int(num_samples * (splits[0] + splits[1]))
        elif split == "test":
            i = int(num_samples * (splits[0] + splits[1]))
            j = num_samples
        else:
            raise ValueError(f"Wrong {split=}")
        return i, j

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data["frame_idx"] = torch.tensor([idx])
        return data
