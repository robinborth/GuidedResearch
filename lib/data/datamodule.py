from functools import partial

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, default_collate

from lib.rasterizer import Rasterizer
from lib.renderer.camera import Camera


class DPHMDataModule(L.LightningDataModule):
    def __init__(
        self,
        # dataset settings
        data_dir: str = "/data",
        # training
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        persistent_workers: bool = False,
        shuffle: bool = True,
        # dataset
        dataset: Dataset | None = None,
        device: str = "cuda",
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.device = device

    def update_dataset(self, camera: Camera, rasterizer: Rasterizer):
        self.scale = camera.scale
        self.dataset = self.hparams["dataset"](camera=camera, rasterizer=rasterizer)
        assert self.hparams["batch_size"] <= self.dataset.optimize_frames

    @staticmethod
    def collate_fn(self, batch: list):
        # move to the device for the dataloader
        _batch: dict = default_collate(batch)
        b = {}
        for key, value in _batch.items():
            if isinstance(value, torch.Tensor):
                b[key] = value.to(self.device)
            else:
                b[key] = value
        return b

    def dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            drop_last=self.hparams["drop_last"],
            persistent_workers=self.hparams["persistent_workers"],
            shuffle=self.hparams["shuffle"],
            collate_fn=partial(self.collate_fn, self),
        )

    def fetch(self):
        return next(iter(self.dataloader()))
