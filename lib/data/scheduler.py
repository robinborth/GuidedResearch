from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import BaseFinetuning, Callback

from lib.model.flame import FLAME


class Scheduler:
    milestones: list[int]

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

    def check_milestones(self, milestones):
        assert len(milestones) >= 1 and milestones[0] == 0


class CoarseToFineScheduler(Callback, Scheduler):
    """Changes the data of the optimization."""

    def __init__(
        self,
        milestones: list[int] = [0],
        scales: list[float] = [1.0],
    ) -> None:
        self.milestones = milestones
        self.check_milestones(milestones)
        self.scales = scales
        self.check_attribute(scales)

    def schedule_dataset(self, trainer: L.Trainer, current_epoch: int):
        if self.skip(current_epoch):
            return
        scale = self.get_attribute(self.scales, current_epoch)
        trainer.datamodule.update_dataset(scale)  # type: ignore

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        self.schedule_dataset(trainer, trainer.current_epoch)

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        self.schedule_dataset(trainer, trainer.current_epoch + 1)


class OptimizerScheduler(Callback, Scheduler):
    """The optimizer scheduler manages to loss mode during optimization."""

    def __init__(
        self,
        milestones: list[int] = [0],
        modes: list[str] = ["default"],
    ) -> None:
        self.milestones = milestones
        self.check_milestones(milestones)
        self.modes = modes
        self.check_attribute(modes)

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


class FinetuneScheduler(BaseFinetuning, Scheduler):
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
        super().__init__()
        self.milestones = milestones
        self.check_milestones(milestones)
        self.params = params
        self.check_attribute(params)

    def iter_modules(self, pl_module: L.LightningModule, current_epoch: int = 0):
        param_string = self.get_attribute(self.params, current_epoch)
        for param in param_string.split("|"):
            yield getattr(pl_module, param), param

    def freeze_before_training(self, pl_module: L.LightningModule) -> None:
        assert isinstance(pl_module, FLAME)
        self.freeze(pl_module)
        for module, param in self.iter_modules(pl_module, 0):
            print("Initilized unfreezed:", param)
            self.make_trainable(module)

    def finetune_function(
        self,
        pl_module: L.LightningModule,
        current_epoch: int,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        if self.skip(current_epoch) or current_epoch == 0:
            return
        assert isinstance(pl_module, FLAME)
        for module, p_name in self.iter_modules(pl_module, current_epoch):
            if p_name in ["shape_params", "expression_params"]:
                lr = 1e-01 if p_name == "shape_params" else 5e-01
                print(f"Unfreeze (step={current_epoch}, lr={lr}): {p_name}")
                self.unfreeze_and_add_param_group(
                    modules=module,
                    optimizer=optimizer,
                    lr=lr,
                )
            else:
                print(f"Unfreeze (step={current_epoch}): {p_name}")
                self.unfreeze_and_add_param_group(
                    modules=module,
                    optimizer=optimizer,
                    initial_denom_lr=1.0,
                )
