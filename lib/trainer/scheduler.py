import logging
from typing import Any

import torch

from lib.data.datamodule import DPHMDataModule
from lib.model.flame import FLAME
from lib.optimizer.base import BaseOptimizer
from lib.optimizer.gd import GradientDecentLinesearch
from lib.optimizer.newton import LevenbergMarquardt
from lib.rasterizer import Rasterizer
from lib.renderer.camera import Camera

log = logging.getLogger()


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
        camera: Camera,
        rasterizer: Rasterizer,
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
        self.camera = camera
        self.rasterizer = rasterizer
        self.prev_scale = 1

    def schedule(self, datamodule: DPHMDataModule, iter_step: int):
        if self.skip(iter_step):
            return
        scale = self.get_attribute(self.scales, iter_step)
        self.prev_scale = scale
        self.camera.update(scale=scale)
        self.rasterizer.update(width=self.camera.width, height=self.camera.height)
        datamodule.update_dataset(camera=self.camera, rasterizer=self.rasterizer)

    # NOTE you can also do a context manager here
    def full_screen(self, datamodule: DPHMDataModule):
        scale = 1
        self.prev_scale = scale
        self.camera.update(scale=scale)
        self.rasterizer.update(width=self.camera.width, height=self.camera.height)
        datamodule.update_dataset(camera=self.camera, rasterizer=self.rasterizer)

    def prev_screen(self, datamodule: DPHMDataModule):
        self.camera.update(scale=self.prev_scale)
        self.rasterizer.update(width=self.camera.width, height=self.camera.height)
        datamodule.update_dataset(camera=self.camera, rasterizer=self.rasterizer)


class OptimizerScheduler(Scheduler):
    """The finetune scheduler manages which flame params are unfreezed.

    This scheduler manages which parameters are frozen in the optimization. One can
    define multiple parameters that can be unfroozen within one milestone. In order to
    do so specify the params with | e.g. "param1|param2" in the params list.
    """

    def __init__(
        self,
        milestones: list[int] = [0],
        params: list[list[str]] = [["global_pose", "transl"]],
        optimizer: str = "levenberg_marquardt",
        optimizer_params: dict[str, Any] | None = None,
        copy_optimizer_state: bool = False,
    ) -> None:
        super().__init__()
        self.milestones = milestones
        self.check_milestones(milestones)

        self.params = params
        self.check_attribute(params)

        self.state: dict[str, Any] = {}
        self.prev_optimizer: Any = None

        # available optimizers
        self.pytorch_optimizers: dict[str, Any] = {
            "adam": torch.optim.Adam,
            "gradient_decent": torch.optim.SGD,
            "gradient_decent_momentum": torch.optim.SGD,
        }
        self.custom_optimizers: dict[str, Any] = {
            "levenberg_marquardt": LevenbergMarquardt,
            "ternary_linesearch": GradientDecentLinesearch,
        }
        self.full_optimizers = {**self.pytorch_optimizers, **self.custom_optimizers}

        # check that the optimizer is correct
        if optimizer not in self.full_optimizers:
            optim_list = list(self.full_optimizers.keys())
            raise ValueError(f"Please select an optimizer from the list: {optim_list}")
        self.optimizer = optimizer

        # check for the configurations
        optimizer_params = {} if optimizer_params is None else optimizer_params
        if self.optimizer == "adam":
            assert "lr" in optimizer_params
        if self.optimizer == "gradient_decent":
            assert "lr" in optimizer_params
            assert not copy_optimizer_state
        if self.optimizer == "gradient_decent_momentum":
            assert "lr" in optimizer_params
            assert "momentum" in optimizer_params
        if self.optimizer == "levenberg_marquardt":
            pass
        if self.optimizer == "ternary_linesearch":
            assert "line_search_fn" in optimizer_params
            assert optimizer_params["line_search_fn"] == "ternary_linesearch"
            assert not copy_optimizer_state
        self.optimizer_params = optimizer_params
        self.copy_optimizer_state = copy_optimizer_state

        self.shared_params = ["shape_params"]

    @property
    def requires_jacobian(self):
        return self.optimizer in ["levenberg_marquardt"]

    @property
    def requires_loss(self):
        return self.optimizer in ["ternary_linesearch"]

    def freeze(self, module: FLAME):
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze(self, module: FLAME):
        for param in module.parameters():
            param.requires_grad = True

    def param_groups(self, model: FLAME, batch: dict, iter_step: int = 0):
        p_names = self.get_attribute(self.params, iter_step=iter_step)
        for p_name in p_names:
            if p_name not in self.state:
                log.info(f"Unfreeze (step={iter_step}): {p_name}")
                module = getattr(model, p_name)

                if p_name in self.shared_params:
                    shape_idx = batch["shape_idx"][:1]
                    params = module(shape_idx)
                else:
                    frame_idx = batch["frame_idx"]
                    params = module(frame_idx)
                params.detach_()
                params.requires_grad_()

                self.state[p_name] = {
                    "params": params,
                    "p_name": p_name,
                }
        return list(self.state.values())

    def get_optimizer(self, model: FLAME, batch: dict, iter_step: int) -> BaseOptimizer:
        param_groups = self.param_groups(model=model, batch=batch, iter_step=iter_step)
        _optimizer = self.full_optimizers[self.optimizer]
        optimizer: BaseOptimizer = _optimizer(
            params=param_groups,
            **self.optimizer_params,
        )  # type: ignore
        return optimizer

    def configure_optimizer(self, model: FLAME, batch: dict, iter_step: int):
        if self.optimizer in self.pytorch_optimizers:
            return self.configure_pytorch_optimizer(model, batch, iter_step)
        if self.optimizer == "levenberg_marquardt":
            return self.configure_levenberg_marquardt(model, batch, iter_step)
        if self.optimizer == "ternary_linesearch":
            return self.configure_ternary_linesearch(model, batch, iter_step)

    def configure_pytorch_optimizer(
        self, model: FLAME, batch: dict, iter_step: int
    ) -> BaseOptimizer:
        optimizer = self.get_optimizer(model, batch, iter_step)
        if self.copy_optimizer_state and self.prev_optimizer is not None:
            optimizer.state = self.prev_optimizer.state
        self.prev_optimizer = optimizer

        # HACK set the state similar to the BaseOptimizer
        _params = []
        _p_names = []
        for group in optimizer.param_groups:
            if len(group["params"]) != 1:
                raise ValueError("Optimizer doesn't support per-parameter options.")
            _params.append(group["params"][0])
            _p_names.append(group["p_name"])
        setattr(optimizer, "_params", _params)
        setattr(optimizer, "_p_names", _p_names)
        setattr(optimizer, "converged", False)

        return optimizer

    def configure_levenberg_marquardt(
        self, model: FLAME, batch: dict, iter_step: int
    ) -> BaseOptimizer:
        optimizer = self.get_optimizer(model, batch, iter_step)
        if self.copy_optimizer_state and self.prev_optimizer is not None:
            optimizer.set_state(self.prev_optimizer.get_state())
        self.prev_optimizer = optimizer
        return optimizer

    def configure_ternary_linesearch(
        self, model: FLAME, batch: dict, iter_step: int
    ) -> BaseOptimizer:
        return self.get_optimizer(model, batch, iter_step)

    def update_model(self, model: FLAME, batch: dict):
        for p_name, group in self.state.items():
            params = group["params"][0]
            module = getattr(model, p_name)
            if p_name in self.shared_params:
                shape_idx = batch["shape_idx"][:1]
                module.weight[shape_idx] = params
            else:
                frame_idx = batch["frame_idx"]
                module.weight[frame_idx] = params
