import logging
from typing import Any

from lib.data.datamodule import DPHMDataModule
from lib.optimizer.base import DifferentiableOptimizer
from lib.renderer import Renderer

log = logging.getLogger()


class Scheduler:
    milestones: list[int]

    def skip(self, iter_step):
        """Skip the scheduler if now new milestone is reached."""
        return not any(iter_step == m for m in self.milestones)

    def set_dirty(self, iter_step):
        self.dirty = not self.skip(iter_step)

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

    def __init__(self, milestones: list[int] = [0], scales: list[int] = [1]) -> None:
        self.milestones = milestones
        self.check_milestones(milestones)
        self.scales = scales
        self.check_attribute(scales)
        for scale in self.scales:
            assert isinstance(scale, int)
            assert scale >= 1

    def schedule(self, datamodule: DPHMDataModule, renderer: Renderer, iter_step: int):
        self.set_dirty(iter_step)
        if self.skip(iter_step):
            return
        scale = self.get_attribute(self.scales, iter_step)
        renderer.update(scale=scale)
        datamodule.update_dataset(
            camera=renderer.camera,
            rasterizer=renderer.rasterizer,
        )


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
        copy_optimizer_state: bool = False,
    ) -> None:
        super().__init__()
        self.milestones = milestones
        self.check_milestones(milestones)

        self.params = params
        self.check_attribute(params)

        self.state: list[str] = []
        self.prev_optimizer_state = None
        self.copy_optimizer_state = copy_optimizer_state

    def p_names(self, iter_step: int = 0):
        p_names = self.get_attribute(self.params, iter_step=iter_step)
        for p_name in p_names:
            if p_name not in self.state:
                self.state.append(p_name)
        return self.state

    def configure_optimizer(self, optimizer: DifferentiableOptimizer, iter_step: int):
        self.set_dirty(iter_step)
        optimizer._p_names = self.p_names(iter_step=iter_step)
