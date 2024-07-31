from typing import Any, Callable

import torch

from lib.optimizer.base import BaseOptimizer


class PytorchOptimizer(BaseOptimizer):
    def __init__(self, optimizer):
        self.optimizer_fn = optimizer
        self.init_step_size = None
        self.step_size = None

    def set_state(self, state):
        self.optimizer.state = state

    def get_state(self):
        return self.optimizer.state

    def set_param_groups(self, param_groups: list[dict[str, Any]]):
        super().set_param_groups(param_groups)
        self.optimizer = self.optimizer_fn(param_groups)

    def step(
        self,
        loss_closure: Callable[[], torch.Tensor],
        jacobian_closure: Callable[[], torch.Tensor],
    ):
        self.optimizer.zero_grad()
        loss = loss_closure()
        loss.backward()
        self.optimizer.step()
