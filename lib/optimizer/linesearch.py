from typing import Callable

import torch

from lib.optimizer.base import DifferentiableOptimizer


def linesearch(
    closure: Callable[[float], float],
    line_search_fn: str | None = None,
) -> float:
    if line_search_fn is None:
        raise ValueError("Currently no line search selected.")
    if line_search_fn == "ternary_search":
        return ternary_search(closure=closure)
    raise ValueError(f"The current {line_search_fn=} is not supported.")


def ternary_search(
    closure: Callable[[float], float],
    ls_a: float = 1e-06,  # the lower bound on the step size
    ls_z: float = 1e06,  # the uppper bound on the step size
    max_steps: int = 25,
) -> float:
    """The ternary search for linesearch.

    Pick some two points b,c such that a<b<c<z.
    If f(b)≤f(c), then x* must be in [a,c]; if f(b)≥(c), then x* must be in [b,z]

    See also:
    https://en.wikipedia.org/wiki/Line_search

    Args:
        evaluate_closure: A function that takes as input the step
            size to evaluate, in the function we compute the objective
            with the provided step size.
        ls_a (float): The lower bound on the step size.
        ls_z (float): The upper bound on the step size.

    Returns:
        float: The optimal step size.
    """
    ls_optim = -1.0

    for _ in range(max_steps):
        # calculate the current learning rates
        ls_delta = ls_z - ls_a
        ls_b = ls_a + 0.49 * ls_delta
        ls_c = ls_a + 0.51 * ls_delta

        # evaluate the points
        fb = closure(ls_b)
        fc = closure(ls_c)

        # update the bounds
        if fb <= fc:  # x* must be in [a,c]
            ls_z = ls_c
            ls_optim = ls_b
        else:  # x* must be in [b,z]
            ls_a = ls_b
            ls_optim = ls_c

    assert ls_optim >= 0.0
    return ls_optim


class GradientDecentLinesearch(DifferentiableOptimizer):
    def step(self, closure: Callable[[dict[str, torch.Tensor]], torch.Tensor]):
        # compute the gradients
        self.zero_grad()
        loss = self.loss_step(closure)
        loss.backward()

        # set the update direction to the negative gradient
        direction = self._gather_flat_grad().neg()

        # prepare the init delta vectors
        x_init = self._clone_param()

        # determine the step size
        step_size = ternary_search(self.evaluate_closure(closure, x_init, direction))

        # update the params
        self._add_direction(step_size=step_size, direction=direction)
