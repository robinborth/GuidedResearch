from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from lib.data.loader import (
    load_color,
    load_depth,
    load_mediapipe_landmark_2d,
    load_mediapipe_landmark_3d,
    load_normal,
)
from lib.renderer import Camera


class DPHMDataset(Dataset):
    def __init__(
        self,
        # rasterizer settings
        camera: Camera,
        # dataset settings
        data_dir: str = "/data",
        sequence_length: int = 1,
        **kwargs,
    ):
        self.sequence_length = sequence_length
        self.camera = camera
        self.data_dir = data_dir
        self.cache_dir = Path(data_dir) / "cache"

    def iter_frame_idx(self):
        yield from range(self.sequence_length)

    def cache_path(self, name: str, frame_idx: int) -> Path:
        return self.cache_dir / f"{self.camera.scale}_{name}" / f"{frame_idx:05}.pt"

    def is_cached(self, name: str, frame_idx: int):
        return self.cache_path(name, frame_idx).exists()

    def load_cached(self, name: str, frame_idx: int):
        path = self.cache_path(name, frame_idx)
        return torch.load(path)

    def save_cached(self, value: torch.Tensor, name: str, frame_idx: int):
        path = self.cache_path(name, frame_idx)
        path.parent.mkdir(parents=True, exist_ok=True)
        return torch.save(value, path)

    def _load_lm_2d_ndc(self, frame_idx):
        landmarks = load_mediapipe_landmark_2d(self.data_dir, idx=frame_idx)
        # the landmarks needs to be in ndc space hence bottom left is (-1,-1)
        landmarks[:, 0] = (landmarks[:, 0] * 2) - 1
        landmarks[:, 1] = -((landmarks[:, 1] * 2) - 1)
        # landmarks[:, 1] = ((landmarks[:, 1] - 1) * 2) + 1
        return landmarks

    def load_lm_2d_ndc(self):
        self.lm_2d_ndc = []
        for frame_idx in self.iter_frame_idx():
            if self.is_cached("lm_2d_ndc", frame_idx):
                landmarks = self.load_cached("lm_2d_ndc", frame_idx)
            else:
                landmarks = self._load_lm_2d_ndc(frame_idx)
                self.save_cached(landmarks, "lm_2d_ndc", frame_idx)
            self.lm_2d_ndc.append(landmarks)

    def _load_lm_3d_camera(self, frame_idx):
        landmarks = load_mediapipe_landmark_3d(self.data_dir, idx=frame_idx)
        landmarks[:, 1] = -landmarks[:, 1]
        landmarks[:, 2] = -landmarks[:, 2]
        return landmarks

    def load_lm_3d_camera(self):
        self.lm_3d_camera = []
        for frame_idx in self.iter_frame_idx():
            if self.is_cached("lm_3d_camera", frame_idx):
                landmarks = self.load_cached("lm_3d_camera", frame_idx)
            else:
                landmarks = self._load_lm_3d_camera(frame_idx)
                self.save_cached(landmarks, "lm_3d_camera", frame_idx)
            self.lm_3d_camera.append(landmarks)

    def _load_color(self, frame_idx: int):
        image = load_color(
            data_dir=self.data_dir,
            idx=frame_idx,
            return_tensor="pt",
        )  # (H, W, 3)
        image = v2.functional.resize(
            inpt=image.permute(2, 0, 1),
            size=(self.camera.height, self.camera.width),
        ).permute(1, 2, 0)
        color = image.to(torch.uint8)  # (H',W',3)
        return color

    def load_color(self):
        self.color = []
        for frame_idx in self.iter_frame_idx():
            if self.is_cached("color", frame_idx):
                _color = self.load_cached("color", frame_idx)
            else:
                _color = self._load_color(frame_idx)
                self.save_cached(_color, "color", frame_idx)
            self.color.append(_color)

    def _load_point(self, frame_idx: int):
        depth = load_depth(
            data_dir=self.data_dir,
            idx=frame_idx,
            return_tensor="pt",
            smooth=True,
        )
        point, mask = self.camera.depth_map_transform(depth)
        return point, mask

    def load_point(self):
        self.point = []
        self.mask = []
        for frame_idx in self.iter_frame_idx():
            if self.is_cached("point", frame_idx) and self.is_cached("mask", frame_idx):
                _point = self.load_cached("point", frame_idx)
                _mask = self.load_cached("mask", frame_idx)
            else:
                _point, _mask = self._load_point(frame_idx)
                self.save_cached(_point, "point", frame_idx)
                self.save_cached(_mask, "mask", frame_idx)
            self.point.append(_point)
            self.mask.append(_mask)

    def _load_normal(self, frame_idx: int):
        normal = load_normal(
            data_dir=self.data_dir,
            idx=frame_idx,
            return_tensor="pt",
            smooth=True,
        )
        normal = v2.functional.resize(
            inpt=normal.permute(2, 0, 1),
            size=(self.camera.height, self.camera.width),
        ).permute(1, 2, 0)
        # to make them right-hand and follow camera space convention +X right +Y up
        # +Z towards the camera
        normal[:, :, 1] = -normal[:, :, 1]
        normal[:, :, 2] = -normal[:, :, 2]
        return normal

    def load_normal(self):
        self.normal = []
        for frame_idx in self.iter_frame_idx():
            if self.is_cached("normal", frame_idx):
                _normal = self.load_cached("normal", frame_idx)
            else:
                _normal = self._load_normal(frame_idx)
                self.save_cached(_normal, "normal", frame_idx)
            self.normal.append(_normal)

    def load_params(self):
        self.params = []
        for frame_idx in self.iter_frame_idx():
            path = Path(self.data_dir) / f"params/{frame_idx:05}.pt"
            _params = torch.load(path)
            _params = {k: v[0] for k, v in _params.items()}
            self.params.append(_params)

    def __len__(self) -> int:
        return self.sequence_length

    def __getitem__(self, idx: int):
        raise NotImplementedError()


class DPHMPointDataset(DPHMDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.load_point()
        self.load_normal()
        self.load_color()
        self.frame_idxs = list(self.iter_frame_idx())

    def __getitem__(self, idx: int):
        # (H', W', 3) this is scaled
        mask = self.mask[idx]
        point = self.point[idx]
        normal = self.normal[idx]
        color = self.color[idx]
        frame_idx = self.frame_idxs[idx]
        return {
            "frame_idx": frame_idx,
            "mask": mask,
            "point": point,
            "normal": normal,
            "color": color,
        }


class DPHMParamsDataset(DPHMDataset):
    def __init__(
        self,
        start_frame: int = 1,
        end_frame: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.load_point()
        self.load_normal()
        self.load_color()
        self.load_params()
        self.frame_idxs = list(self.iter_frame_idx())
        self.start_frame = start_frame
        self.end_frame = end_frame

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
        params = self.params[frame_idx - 1]
        init_color = self.color[frame_idx - 1]
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
