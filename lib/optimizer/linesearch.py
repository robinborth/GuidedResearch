from typing import Callable

import torch

from lib.optimizer.base import BaseOptimizer


class GradientDecentLinesearch(BaseOptimizer):

    @torch.no_grad()
    def step(
        self,
        loss_closure: Callable[[], torch.Tensor],
        jacobian_closure: Callable[[], torch.Tensor],
    ) -> float:
        # compute the gradients
        with torch.enable_grad():
            self.zero_grad()
            loss = loss_closure()
            loss.backward()
            direction = self._gather_flat_grad().neg()  # we minimize

        # prepare the init delta vectors
        x_init = self._clone_param()

        # determine the step size
        step_size = self.lr
        if self.perform_linesearch:
            step_size = self.linesearch(
                loss_closure=loss_closure,
                x_init=x_init,
                direction=direction,
            )

        self._add_direction(step_size=step_size, direction=direction)
        return float(loss)
