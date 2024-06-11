from typing import Any

import torch

from lib.data.datamodule import DPHMDataModule
from lib.model.flame import FLAME


class Scheduler:
    milestones: list[int]

    def skip(self, current_epoch):
        """Skip the scheduler if now new milestone is reached."""
        return not any(current_epoch == m for m in self.milestones)

    def get_attribute(self, attributes: list[Any], iter_step: int):
        """Select the current attribute from the list."""
        assert iter_step >= 0
        if self.milestones is None or len(self.milestones) == 0:
            return None
        milestone_idx = sum(m <= iter_step for m in self.milestones) - 1
        return attributes[milestone_idx]

    def check_attribute(self, attribute):
        assert len(attribute) == len(self.milestones)

    def check_milestones(self, milestones):
        assert len(milestones) >= 1 and milestones[0] == 0


class CoarseToFineScheduler(Scheduler):
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

    def schedule_dataset(self, datamodule: DPHMDataModule, iter_step: int):
        if self.skip(iter_step):
            return
        scale = self.get_attribute(self.scales, iter_step)
        datamodule.update_dataset(scale)


class FinetuneScheduler(Scheduler):
    """The finetune scheduler manages which flame params are unfreezed.

    This scheduler manages which parameters are frozen in the optimization. One can
    define multiple parameters that can be unfroozen within one milestone. In order to
    do so specify the params with | e.g. "param1|param2" in the params list.
    """

    def __init__(
        self,
        milestones: list[int] = [0],
        params: list[list[str]] = [["global_pose", "transl"]],
        lr: list[list[float]] = [[1e-02, 1e-02]],
    ) -> None:
        super().__init__()
        self.milestones = milestones
        self.check_milestones(milestones)
        self.params = params
        self.check_attribute(params)
        self.lr = lr
        self.check_attribute(lr)
        self.state: dict[str, Any] = {}

    def freeze(self, module: FLAME):
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze(self, module: FLAME):
        for param in module.parameters():
            param.requires_grad = True

    def update_state(self, iter_step: int = 0):
        params = self.get_attribute(self.params, iter_step=iter_step)
        lrs = self.get_attribute(self.lr, iter_step=iter_step)
        for param, lr in zip(params, lrs):
            print(f"Unfreeze (step={iter_step}, lr={lr}): {param}")
            self.state[param] = {"param": param, "lr": lr}

    def param_groups(self, model: FLAME, iter_step: int = 0):
        self.update_state(iter_step)
        groups = []
        for s in self.state.values():
            parameters = getattr(model, s["param"]).parameters()
            groups.append({"params": parameters, "lr": s["lr"]})
        return groups

    def configure_optimizers(
        self, model: FLAME, iter_step: int
    ) -> torch.optim.Optimizer:
        params = self.param_groups(model=model, iter_step=iter_step)
        return torch.optim.Adam(params=params)


class DummyScheduler(Scheduler):
    def __getattr__(self, name):
        # This is a no-operation function
        def no_op(*args, **kwargs):
            pass

        return no_op
