import logging
from typing import Any

from lib.data.datamodule import DPHMDataModule
from lib.model.flame import FLAME
from lib.optimizer.base import BaseOptimizer
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

    def reset(self):
        pass


class CoarseToFineScheduler(Scheduler):
    """Changes the data of the optimization."""

    def __init__(self, milestones: list[int] = [0], scales: list[int] = [1]) -> None:
        self.milestones = milestones
        self.check_milestones(milestones)
        self.scales = scales
        self.check_attribute(scales)
        for scale in self.scales:
            assert isinstance(scale, int)
            assert scale >= 1
        self.prev_scale = 1

    def init_scheduler(self, camera: Camera, rasterizer: Rasterizer):
        self.camera = camera
        self.rasterizer = rasterizer

    def schedule(self, datamodule: DPHMDataModule, iter_step: int):
        if self.skip(iter_step):
            return
        scale = self.get_attribute(self.scales, iter_step)
        self.prev_scale = scale
        self.camera.update(scale=scale)
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
        # optimizer: str = "levenberg_marquardt",
        # optimizer_params: dict[str, Any] | None = None,
        copy_optimizer_state: bool = False,
    ) -> None:
        super().__init__()
        self.milestones = milestones
        self.check_milestones(milestones)

        self.params = params
        self.check_attribute(params)

        self.state: dict[str, Any] = {}
        self.prev_optimizer_state = None
        self.copy_optimizer_state = copy_optimizer_state

    def freeze(self, module: FLAME):
        for param in module.parameters():
            param.detach_()
            param.requires_grad_(False)

    def param_groups(self, model: FLAME, batch: dict, iter_step: int = 0):
        p_names = self.get_attribute(self.params, iter_step=iter_step)
        for p_name in p_names:
            if p_name not in self.state:
                # log.info(f"Unfreeze (step={iter_step}): {p_name}")
                module = getattr(model, p_name)

                if p_name in model.shape_p_names:
                    shape_idx = batch["shape_idx"][:1]
                    params = module(shape_idx)
                else:
                    frame_idx = batch["frame_idx"]
                    params = module(frame_idx)
                params.detach_()
                params.requires_grad_()

                self.state[p_name] = {
                    "params": [params],
                    "p_name": p_name,
                }
        return list(self.state.values())

    def configure_optimizer(
        self,
        optimizer: BaseOptimizer,
        model: FLAME,
        batch: dict,
        iter_step: int,
    ):
        param_groups = self.param_groups(model, batch, iter_step)
        optimizer.set_param_groups(param_groups)
        optimizer.reset()
        if self.copy_optimizer_state and self.prev_optimizer_state is not None:
            optimizer.set_state(self.prev_optimizer_state)
        self.prev_optimizer_state = optimizer.get_state()

    def update_model(self, model: FLAME, batch: dict):
        for p_name, group in self.state.items():
            params = group["params"][0]
            module = getattr(model, p_name)
            if p_name in model.shape_p_names:
                shape_idx = batch["shape_idx"][:1]
                module.weight[shape_idx] = params
            else:
                frame_idx = batch["frame_idx"]
                module.weight[frame_idx] = params

    def reset(self):
        self.state = {}
        self.prev_optimizer_state = None
