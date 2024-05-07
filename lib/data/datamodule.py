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
        image_scales: list[float] = [1.0],
        milestones: list[int] = [0],
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

        self.image_scales = image_scales
        self.milestones = milestones

        self.base_width = image_width
        self.base_height = image_height

    def update_dataset(self):
        current_epoch = self.trainer.current_epoch
        milestone_idx = sum(m <= current_epoch for m in self.milestones) - 1

        self.image_scale = self.image_scales[milestone_idx]
        self.image_height = int(self.image_scale * self.base_height)
        self.image_width = int(self.image_scale * self.base_width)
        self.dataset = self.datasets[milestone_idx]

    def load_dataset(self, milestone_idx: int):
        image_scale = self.image_scales[milestone_idx]
        image_height = int(image_scale * self.base_height)
        image_width = int(image_scale * self.base_width)
        return self.hparams["dataset"](
            image_scale=image_scale,
            image_width=image_width,
            image_height=image_height,
        )

    def setup(self, stage: str) -> None:
        self.datasets = []
        for idx, _ in enumerate(self.milestones):
            dataset = self.load_dataset(idx)
            self.datasets.append(dataset)

    def train_dataloader(self) -> DataLoader:
        self.update_dataset()
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            drop_last=self.hparams["drop_last"],
            persistent_workers=self.hparams["persistent_workers"],
            shuffle=self.hparams["shuffle"],
        )
