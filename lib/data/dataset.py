import random
from pathlib import Path

import torch
from torch.utils.data import Dataset


class DPHMDataset(Dataset):
    def __init__(self, scale: int = 1, data_dir: str = "/data"):
        self.scale = scale
        self.data_dir = data_dir

    def frame_count(self, dataset: str):
        path = Path(self.data_dir) / dataset / "cache/8_mask"
        return len([p for p in path.iterdir() if str(p).endswith(".pt")])

    def iter_frame_idx(self, dataset: str):
        return list(range(self.frame_count(dataset)))

    def load_cached(self, dataset: str, data_type: str, frame_idx: int) -> Path:
        path = (
            Path(self.data_dir)
            / dataset
            / "cache"
            / f"{self.scale}_{data_type}"
            / f"{frame_idx:05}.pt"
        )
        return torch.load(path)

    def load(self, dataset: str, data_type: str):
        data = []
        for frame_idx in self.iter_frame_idx(dataset):
            value = self.load_cached(dataset, data_type, frame_idx)
            data.append(value)
        return data

    def load_root(self, dataset: str, data_type: str, frame_idx: int):
        path = Path(self.data_dir) / f"{dataset}/{data_type}/{frame_idx:05}.pt"
        return torch.load(path)

    def load_roots(self, dataset: str, data_type: str):
        data = []
        for frame_idx in self.iter_frame_idx(dataset):
            value = self.load_root(dataset, data_type, frame_idx)
            data.append(value)
        return data

    def load_param(self, dataset: str, frame_idx: int):
        path = Path(self.data_dir) / f"{dataset}/params/{frame_idx:05}.pt"
        params = torch.load(path)
        return {k: v[0] for k, v in params.items()}

    def load_params(self, dataset: str):
        params = []
        for frame_idx in self.iter_frame_idx(dataset):
            params.append(self.load_param(dataset, frame_idx))
        return params

    def __len__(self) -> int:
        raise NotImplementedError()

    def __getitem__(self, idx: int):
        raise NotImplementedError()


class DPHMOptimizeDataset(DPHMDataset):
    def __init__(
        self,
        dataset: str,
        scale: int = 1,
        data_dir: str = "/data",
    ):
        self.scale = scale
        self.data_dir = data_dir
        self.dataset = dataset
        self.mask = self.load(dataset, "mask")
        self.normal = self.load(dataset, "normal")
        self.color = self.load(dataset, "color")
        self.point = self.load(dataset, "point")
        self.landmark = self.load_roots(dataset, "landmark")
        self.landmark_mask = self.load_roots(dataset, "landmark_mask")
        self.vertices = self.load_roots(dataset, "vertices")
        self.frame_idxs = self.iter_frame_idx(dataset)

    def __len__(self) -> int:
        return self.frame_count(self.dataset)

    def __getitem__(self, idx: int):
        mask = self.mask[idx]
        point = self.point[idx]
        normal = self.normal[idx]
        color = self.color[idx]
        landmark = self.landmark[idx]
        landmark_mask = self.landmark_mask[idx]
        frame_idx = self.frame_idxs[idx]
        # (H', W', 3) this is scaled
        return {
            "frame_idx": frame_idx,
            "mask": mask,
            "point": point,
            "normal": normal,
            "color": color,
            "landmark": landmark,
            "landmark_mask": landmark_mask,
        }


class DPHMTrainDataset(DPHMDataset):
    def __init__(
        self,
        scale: int = 1,
        data_dir: str = "/data",
        datasets: list[str] = [],
        mode: str = "fix",
        jump_size: int = 1,
        start_frame: int | None = None,
        end_frame: int | None = None,
        memory: str = "ram",  # ram, disk
        landmarks: bool = True,
        **kwargs,
    ):
        assert mode in ["fix", "dynamic"]
        self.mode = mode
        self.jump_size = jump_size
        self.scale = scale
        self.memory = memory
        self.data_dir = data_dir
        self._start_frame = start_frame
        self._end_frame = end_frame

        self.total_frames = 0
        self.mask = {}
        self.face_mask = {}
        self.normal = {}
        self.color = {}
        self.point = {}
        self.params = {}
        self.vertices = {}
        self.landmark = {}
        self.landmark_mask = {}
        self.frame_idx = {}
        self.start_frame = {}
        self.start_frame_idx = {}
        self.end_frame = {}
        self.idx2dataset = {}
        self.landmarks_flag = landmarks

        for dataset in sorted(datasets):
            if self.memory == "ram":
                self.mask[dataset] = self.load(dataset, "mask")
                self.face_mask[dataset] = self.load(dataset, "face_mask")
                self.normal[dataset] = self.load(dataset, "normal")
                self.color[dataset] = self.load(dataset, "color")
                self.point[dataset] = self.load(dataset, "point")
                self.params[dataset] = self.load_params(dataset)
                self.vertices[dataset] = self.load_roots(dataset, "vertices")
                if self.landmarks_flag:
                    self.landmark[dataset] = self.load_roots(dataset, "landmark")
                    self.landmark_mask[dataset] = self.load_roots(
                        dataset, "landmark_mask"
                    )

            start_frame = self._start_frame
            if start_frame is None:
                start_frame = self.jump_size
            assert start_frame >= self.jump_size
            self.start_frame[dataset] = start_frame
            self.start_frame_idx[dataset] = self.total_frames

            end_frame = self._end_frame
            if end_frame is None:
                end_frame = self.frame_count(dataset)
                if self.mode == "dynamic":
                    end_frame -= self.jump_size
            assert end_frame <= self.frame_count(dataset)
            self.end_frame[dataset] = end_frame

            frame_idx = list(range(start_frame, end_frame))
            self.frame_idx[dataset] = frame_idx

            idx2data = {self.total_frames + i: dataset for i in range(len(frame_idx))}
            self.idx2dataset.update(idx2data)
            self.total_frames += len(frame_idx)

    def fetch_helper(self, idx: int):
        # fetch the information about the current dataset sequence
        dataset = self.idx2dataset[idx]
        start_frame = self.start_frame[dataset]
        start_frame_idx = self.start_frame_idx[dataset]

        # sequence indexing
        frame_idx = (idx - start_frame_idx) + start_frame
        init_idx = frame_idx - self.jump_size
        if self.mode == "dynamic":
            s_idx = frame_idx - self.jump_size
            e_idx = frame_idx + self.jump_size
            init_idx = random.randint(s_idx, e_idx)

        return dataset, frame_idx, init_idx

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx: int):
        dataset, frame_idx, init_idx = self.fetch_helper(idx)
        landmark = []
        landmark_mask = []
        if self.memory == "ram":
            mask = self.mask[dataset][frame_idx]
            # face_mask = self.face_mask[dataset][frame_idx]
            point = self.point[dataset][frame_idx]
            normal = self.normal[dataset][frame_idx]
            color = self.color[dataset][frame_idx]
            params = self.params[dataset][frame_idx]
            vertices = self.vertices[dataset][frame_idx]
            init_params = self.params[dataset][init_idx]
            init_vertices = self.vertices[dataset][init_idx]
            init_color = self.color[dataset][init_idx]
            if self.landmarks_flag:
                landmark = self.landmark[dataset][frame_idx]
                landmark_mask = self.landmark_mask[dataset][frame_idx]
        else:
            mask = self.load_cached(dataset, "mask", frame_idx)
            # face_mask = self.load_cached(dataset, "face_mask", frame_idx)
            point = self.load_cached(dataset, "point", frame_idx)
            normal = self.load_cached(dataset, "normal", frame_idx)
            color = self.load_cached(dataset, "color", frame_idx)
            params = self.load_param(dataset, frame_idx)
            vertices = self.load_root(dataset, "vertices", frame_idx)
            init_params = self.load_param(dataset, init_idx)
            init_vertices = self.load_root(dataset, "vertices", init_idx)
            init_color = self.load_cached(dataset, "color", init_idx)
            if self.landmarks_flag:
                landmark = self.load_root(dataset, "landmark", frame_idx)
                landmark_mask = self.load_root(dataset, "landmark_mask", frame_idx)
        return {
            "dataset": dataset,
            "frame_idx": frame_idx,
            "mask": mask,
            # "face_mask": face_mask,
            "point": point,
            "normal": normal,
            "color": color,
            "params": params,
            "vertices": vertices,
            "init_params": init_params,
            "init_vertices": init_vertices,
            "init_color": init_color,
            "init_frame_idx": frame_idx,
            "landmark": landmark,
            "landmark_mask": landmark_mask,
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
