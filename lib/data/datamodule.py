import lightning as L
from torch.utils.data import DataLoader

from lib.data.dataset import DPHMPointCloudDataset


class DPHMPointCloudDataModule(L.LightningDataModule):
    def __init__(
        self,
        # dataset settings
        data_dir: str = "/data",
        chunk_size: int = 50000,
        num_frames: int = 1,
        start_frame_idx: int = 0,
        # training
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        persistent_workers: bool = False,
        shuffle: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        assert batch_size <= num_frames

    def setup(self, stage: str) -> None:
        self.dataset = DPHMPointCloudDataset(
            data_dir=self.hparams["data_dir"],
            chunk_size=self.hparams["chunk_size"],
            num_frames=self.hparams["num_frames"],
            start_frame_idx=self.hparams["start_frame_idx"],
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
