import lightning as L
from torch.utils.data import DataLoader, Dataset


class DPHMDataModule(L.LightningDataModule):
    def __init__(
        self,
        # dataset settings
        data_dir: str = "/data",
        optimize_frames: int = 1,
        start_frame_idx: int = 0,
        # rasterizer settings
        image_scale: int = 1,
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
        assert batch_size <= optimize_frames

        self.image_scale = image_scale
        self.image_width = int(image_width * image_scale)
        self.image_height = int(image_height * image_scale)

    def setup(self, stage: str) -> None:
        self.dataset = self.hparams["dataset"](
            image_scale=self.image_scale,
            image_width=self.image_width,
            image_height=self.image_height,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            drop_last=self.hparams["drop_last"],
            persistent_workers=self.hparams["persistent_workers"],
            shuffle=self.hparams["shuffle"],
        )
