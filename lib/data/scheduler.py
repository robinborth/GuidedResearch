from typing import Any

import torch

from lib.data.datamodule import DPHMDataModule
from lib.model.flame import FLAME
from lib.optimizer.jacobian import FlameJacobian
from lib.optimizer.newton import LevenbergMarquardt


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
        scales: list[int] = [1],
    ) -> None:
        self.milestones = milestones
        self.check_milestones(milestones)
        self.scales = scales
        self.check_attribute(scales)
        for scale in self.scales:
            assert isinstance(scale, int)
            assert scale >= 1

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
        self.prev_optimizer: None | torch.optim.Optimizer = None

    def freeze(self, module: FLAME):
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze(self, module: FLAME):
        for param in module.parameters():
            param.requires_grad = True

    def param_groups(self, model: FLAME, iter_step: int = 0):
        params = self.get_attribute(self.params, iter_step=iter_step)
        lrs = self.get_attribute(self.lr, iter_step=iter_step)
        for param, lr in zip(params, lrs):
            if param not in self.state:
                print(f"Unfreeze (step={iter_step}, lr={lr}): {param}")
                module = getattr(model, param)
                self.unfreeze(module)
                self.state[param] = {
                    "params": module.parameters(),
                    "lr": lr,
                    "p_name": param,
                }
        return list(self.state.values())

    def configure_adam(
        self, model: FLAME, iter_step: int, copy_state: bool = False, **kwargs
    ) -> torch.optim.Optimizer:
        param_groups = self.param_groups(model=model, iter_step=iter_step)
        optimizer = torch.optim.Adam(params=param_groups)
        if copy_state and self.prev_optimizer is not None:
            optimizer.state = self.prev_optimizer.state
        self.prev_optimizer = optimizer
        setattr(optimizer, "converged", False)
        return optimizer

    def configure_levenberg_marquardt(
        self,
        model: FLAME,
        batch: dict,
        correspondences: dict,
        iter_step: int,
        copy_state: bool = False,
        **kwargs,
    ) -> torch.optim.Optimizer:
        param_groups = self.param_groups(model=model, iter_step=iter_step)
        params = [group["p_name"] for group in param_groups]
        jacobian = FlameJacobian(
            model=model,
            batch=batch,
            correspondences=correspondences,
            params=params,
        )
        optimizer = LevenbergMarquardt(jacobian=jacobian, params=param_groups)
        if copy_state and self.prev_optimizer is not None:
            prev_damping_factor = self.prev_optimizer.damping_factor  # type: ignore
            optimizer.damping_factor = prev_damping_factor
        self.prev_optimizer = optimizer
        return optimizer

    def requires_jacobian(self, optimizer):
        return isinstance(optimizer, LevenbergMarquardt)
