from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import BaseFinetuning, Callback

from lib.model.flame import FLAME


class Scheduler:
    def __init__(self, milestones: list[int] = []):
        self.milestones = milestones

    def skip(self, current_epoch):
        """Skip the scheduler if now new milestone is reached."""
        return not any(current_epoch == m for m in self.milestones)

    def get_attribute(self, attributes: list[Any], current_epoch: int):
        """Select the current attribute from the list."""
        assert current_epoch >= 0
        if self.milestones is None or len(self.milestones) == 0:
            return None
        milestone_idx = sum(m <= current_epoch for m in self.milestones) - 1
        return attributes[milestone_idx]

    def check_attribute(self, attribute):
        assert len(attribute) == len(self.milestones)


class CoarseToFineScheduler(Scheduler, Callback):
    """Changes the data of the optimization."""

    def __init__(
        self,
        milestones: list[int] = [0],
        image_scales: list[float] = [1.0],
    ) -> None:
        super().__init__(milestones)
        self.check_attribute(image_scales)
        self.image_scales = image_scales

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if self.skip(trainer.current_epoch):
            return
        # override the rendering settings in the datamodule, which implictily sets the
        # scale, width, height in the flame model, by reference.
        image_scale = self.get_attribute(self.image_scales, trainer.current_epoch)
        trainer.datamodule.set_dataset(image_scale)  # type: ignore


class OptimizerScheduler(Scheduler, Callback):
    """The optimizer scheduler manages to loss mode during optimization."""

    def __init__(
        self,
        milestones: list[int] = [0],
        modes: list[str] = ["default"],
    ) -> None:
        super().__init__(milestones)
        self.check_attribute(modes)
        self.modes = modes

    def on_train_epoch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        if self.skip(trainer.current_epoch):
            return
        assert isinstance(pl_module, FLAME)
        mode = self.get_attribute(self.modes, trainer.current_epoch)
        pl_module.set_optimize_mode(mode)


class FinetuneScheduler(Scheduler, BaseFinetuning):
    """The finetune scheduler manages which flame params are unfreezed.

    This scheduler manages which parameters are frozen in the optimization. One can
    define multiple parameters that can be unfroozen within one milestone. In order to
    do so specify the params with | e.g. "param1|param2" in the params list.
    """

    def __init__(
        self,
        milestones: list[int] = [0],
        params: list[str] = ["global_pose|transl"],
    ) -> None:
        super().__init__(milestones)
        self.check_attribute(params)
        self.params = params

    def freeze_before_training(self, pl_module: L.LightningModule) -> None:
        assert isinstance(pl_module, FLAME)
        self.freeze(pl_module)

    def finetune_function(
        self,
        pl_module: L.LightningModule,
        current_epoch: int,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        if self.skip(current_epoch):
            return
        assert isinstance(pl_module, FLAME)
        param_string = self.get_attribute(self.params, current_epoch)
        for param in param_string.split("|"):
            modules = getattr(pl_module, param)
            self.unfreeze_and_add_param_group(modules=modules, optimizer=optimizer)
