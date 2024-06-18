from functools import partial

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, default_collate

from lib.data.loader import load_intrinsics
from lib.rasterizer import Rasterizer
from lib.renderer.camera import Camera


class DPHMDataModule(L.LightningDataModule):
    def __init__(
        self,
        # dataset settings
        data_dir: str = "/data",
        # rasterizer settings
        scale: int = 1,
        width: int = 1920,
        height: int = 1080,
        near: float = 0.01,
        far: float = 100,
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
        self.scale = scale
        self.device = device

    def update_dataset(self, scale: int = 1):
        self.scale = scale
        self.camera.update(scale=self.scale)
        self.rasterizer.update(width=self.camera.width, height=self.camera.height)
        self.dataset = self.hparams["dataset"](
            camera=self.camera,
            rasterizer=self.rasterizer,
        )
        assert self.hparams["batch_size"] <= self.dataset.optimize_frames

    def setup(self):
        self.K = load_intrinsics(
            data_dir=self.hparams["data_dir"],
            return_tensor="pt",
        )
        self.camera = Camera(
            K=self.K,
            scale=self.scale,  # this does not matter we update it in update dataset
            width=self.hparams["width"],
            height=self.hparams["height"],
            near=self.hparams["near"],
            far=self.hparams["far"],
        )
        self.rasterizer = Rasterizer(
            width=self.camera.width,
            height=self.camera.height,
        )

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

    def train_dataloader(self) -> DataLoader:
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
