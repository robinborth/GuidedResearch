import time
from typing import Any, Callable

import torch

from lib.optimizer.base import DifferentiableOptimizer


class PytorchOptimizer(DifferentiableOptimizer):
    def __init__(self, optimizer: Callable[..., torch.optim.Optimizer]):
        super().__init__()
        self.optimizer_fn = optimizer
        self.init_step_size = None
        self.step_size = None

    def set_state(self, state):
        self.optimizer.state = state

    def get_state(self):
        return self.optimizer.state

    def set_params(self, params: dict[str, Any]):
        super().set_params(params)
        param_groups = [{"params": p} for p in params.values()]
        self.optimizer = self.optimizer_fn(param_groups)

    def step(self, closure: Callable[[dict[str, torch.Tensor]], torch.Tensor]):
        self.optimizer.zero_grad()
        loss = self.loss_step(closure)
        loss.backward()
        self.optimizer.step()
