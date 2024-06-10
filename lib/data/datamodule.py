import lightning as L
from torch.utils.data import DataLoader, Dataset

from lib.rasterizer import Rasterizer
from lib.renderer.camera import Camera
from lib.utils.loader import load_intrinsics


class DPHMDataModule(L.LightningDataModule):
    def __init__(
        self,
        # dataset settings
        data_dir: str = "/data",
        # rasterizer settings
        scale: float = 1.0,
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
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.scale = scale

    def update_dataset(self, scale: float = 1.0):
        self.scale = scale
        self.camera.update(scale=self.scale)
        self.rasterizer.update(width=self.camera.width, height=self.camera.height)
        self.dataset = self.hparams["dataset"](
            camera=self.camera,
            rasterizer=self.rasterizer,
        )
        assert self.hparams["batch_size"] <= self.dataset.optimize_frames

    def setup(self, stage: str = ""):
        self.K = load_intrinsics(
            data_dir=self.hparams["data_dir"],
            return_tensor="pt",
        )
        self.camera = Camera(
            K=self.K,
            scale=self.hparams["scale"],
            width=self.hparams["width"],
            height=self.hparams["height"],
            near=self.hparams["near"],
            far=self.hparams["far"],
        )
        self.rasterizer = Rasterizer(
            width=self.camera.width,
            height=self.camera.height,
        )

    def train_dataloader(self) -> DataLoader:
        # default settings if coarse to fine scheduler is not set, note that if one
        # want's to use multiple gpu's this should be moved to setup.
        if not hasattr(self, "dataset"):
            self.update_dataset(self.scale)

        return DataLoader(
            dataset=self.dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            drop_last=self.hparams["drop_last"],
            persistent_workers=self.hparams["persistent_workers"],
            shuffle=self.hparams["shuffle"],
        )
