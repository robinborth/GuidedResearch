from typing import Callable

import torch

from lib.optimizer.base import DifferentiableOptimizer


class GradientDecent(DifferentiableOptimizer):
    def __init__(self, step_size: float = 1.0):
        super().__init__()
        self.step_size = step_size

    def step(self, closure: Callable[[dict[str, torch.Tensor]], torch.Tensor]):
        # compute the gradients
        self.zero_grad()
        loss = self.loss_step(closure)
        loss.backward(create_graph=True)
        # set the update direction to the negative gradient
        direction = self._gather_flat_grad().neg()
        # update the params
        self._add_direction(step_size=self.step_size, direction=direction)
