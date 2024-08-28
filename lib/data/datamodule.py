from functools import partial

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, default_collate

from lib.data.sampler import SimpleIndexSampler
from lib.renderer import Camera, Rasterizer, Renderer


# Dataset for optimization
class DPHMDataModule(L.LightningDataModule):
    def __init__(
        self,
        # dataset settings
        data_dir: str = "/data",
        # training
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        # dataset
        dataset: Dataset | None = None,
        device: str = "cuda",
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.device = device
        self.batch_size = batch_size
        self.sampler: None | SimpleIndexSampler = None
        self._datasets: dict[int, torch.utils.data.Dataset] = {}

    @staticmethod
    def _collate_fn(self, batch: list):
        # move to the device for the dataloader
        _batch: dict = default_collate(batch)
        b = {}
        for key, value in _batch.items():
            if isinstance(value, torch.Tensor):
                b[key] = value.to(self.device)
            else:
                b[key] = value
        return b

    def update_dataset(self, camera: Camera, rasterizer: Rasterizer):
        """This is modified by the coarse to fine scheduler."""
        self.scale = camera.scale
        if self.scale not in self._datasets:  # cache the dataset with the scale
            _dataset = self.hparams["dataset"](camera=camera, rasterizer=rasterizer)
            self._datasets[self.scale] = _dataset
        self.dataset = self._datasets[self.scale]  # select the dataset with the scale

    def update_idxs(self, idxs: list[int]):
        """This is used to change the sampling mode of the datasets."""
        self.sampler = SimpleIndexSampler(idxs)
        self.batch_size = len(idxs)

    def fetch(self):
        assert self.sampler is not None
        dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            persistent_workers=self.hparams["persistent_workers"],
            collate_fn=partial(self._collate_fn, self),
            sampler=self.sampler,
        )
        return next(iter(dataloader))


class SplitDataModule(L.LightningDataModule):
    def __init__(
        self,
        # dataset settings
        data_dir: str = "/data",
        # training
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = True,
        persistent_workers: bool = False,
        shuffle: bool = True,
        # dataset
        dataset: Dataset | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

    def setup(self, stage: str):
        if stage in ["fit", "all"]:
            self.train_dataset = self.hparams["dataset"](split="train")
        if stage in ["validate", "fit", "all"]:
            self.val_dataset = self.hparams["dataset"](split="val")
        if stage in ["test", "all"]:
            self.test_dataset = self.hparams["dataset"](split="test")
        if stage in ["predict", "all"]:
            self.predict_dataset = self.hparams["dataset"](split="val")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            drop_last=self.hparams["drop_last"],
            persistent_workers=self.hparams["persistent_workers"],
            shuffle=self.hparams["shuffle"],
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            drop_last=self.hparams["drop_last"],
            persistent_workers=self.hparams["persistent_workers"],
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            drop_last=self.hparams["drop_last"],
            persistent_workers=self.hparams["persistent_workers"],
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.predict_dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            drop_last=self.hparams["drop_last"],
            persistent_workers=self.hparams["persistent_workers"],
        )


class SyntheticDataModule(L.LightningDataModule):
    def __init__(
        self,
        # dataset settings
        data_dir: str = "/data",
        split: list[float] = [0.8, 0.1, 0.1],
        # training
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = True,
        persistent_workers: bool = False,
        shuffle: bool = True,
        # dataset
        dataset: Dataset | None = None,
        renderer: Renderer | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["renderer"])
        self.renderer = renderer

    def setup(self, stage: str):
        if stage in ["fit", "all"]:
            self.train_dataset = self.hparams["dataset"](split="train")
        if stage in ["validate", "fit", "all"]:
            self.val_dataset = self.hparams["dataset"](split="val")
        if stage in ["test", "all"]:
            self.test_dataset = self.hparams["dataset"](split="test")
        if stage in ["predict", "all"]:
            self.predict_dataset = self.hparams["dataset"](split="val")

    def collate_fn(self, batch):
        assert len(batch) == 1
        return batch[0]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=1,
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            drop_last=self.hparams["drop_last"],
            persistent_workers=self.hparams["persistent_workers"],
            shuffle=self.hparams["shuffle"],
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=1,
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            drop_last=self.hparams["drop_last"],
            persistent_workers=self.hparams["persistent_workers"],
            collate_fn=self.collate_fn,
        )


class KinectDataModule(L.LightningDataModule):
    def __init__(
        self,
        # dataset settings
        data_dir: str = "/data",
        split: list[float] = [0.8, 0.1, 0.1],
        # training
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = True,
        persistent_workers: bool = False,
        shuffle: bool = True,
        # dataset
        dataset: Dataset | None = None,
        renderer: Renderer | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["renderer"])
        self.renderer = renderer

    def setup(self, stage: str = "all"):
        camera = self.renderer.camera  # type: ignore
        self.dataset = self.hparams["dataset"](camera=camera)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset,
            batch_size=1,
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            drop_last=self.hparams["drop_last"],
            persistent_workers=self.hparams["persistent_workers"],
            shuffle=self.hparams["shuffle"],
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset,
            batch_size=1,
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            drop_last=self.hparams["drop_last"],
            persistent_workers=self.hparams["persistent_workers"],
            shuffle=False,
        )
