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
        self.base_scale = image_scale
        self.base_width = image_width
        self.base_height = image_height

    def prepare_dataset(self, image_scale):
        self.image_scale = image_scale
        self.image_height = int(self.image_scale * self.base_height)
        self.image_width = int(self.image_scale * self.base_width)
        self.dataset = self.hparams["dataset"](
            image_scale=self.image_scale,
            image_width=self.image_width,
            image_height=self.image_height,
        )
        assert self.hparams["batch_size"] <= self.dataset.optimize_frames

    def train_dataloader(self) -> DataLoader:
        # default settings if coarse to fine scheduler is not set, note that if one
        # want's to use multiple gpu's this should be moved to setup.
        if not hasattr(self, "dataset"):
            self.prepare_dataset(self.base_scale)

        return DataLoader(
            dataset=self.dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            drop_last=self.hparams["drop_last"],
            persistent_workers=self.hparams["persistent_workers"],
            shuffle=self.hparams["shuffle"],
        )
