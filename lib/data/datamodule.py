import lightning as L
from torch.utils.data import DataLoader, Dataset


class DPHMDataModule(L.LightningDataModule):
    def __init__(
        self,
        # dataset settings
        data_dir: str = "/data",
        # rasterizer settings
        image_scale: float = 1.0,
        image_width: int = 1920,
        image_height: int = 1080,
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
        self.scale = image_scale
        self.width = image_width
        self.height = image_height

    def prepare_dataset(self, scale: float = 1.0):
        self.scale = scale
        self.dataset = self.hparams["dataset"](
            image_scale=self.scale,
            image_width=self.width,
            image_height=self.height,
        )
        assert self.hparams["batch_size"] <= self.dataset.optimize_frames

    def train_dataloader(self) -> DataLoader:
        # default settings if coarse to fine scheduler is not set, note that if one
        # want's to use multiple gpu's this should be moved to setup.
        if not hasattr(self, "dataset"):
            self.prepare_dataset(self.scale)

        return DataLoader(
            dataset=self.dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            drop_last=self.hparams["drop_last"],
            persistent_workers=self.hparams["persistent_workers"],
            shuffle=self.hparams["shuffle"],
        )
