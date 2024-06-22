from typing import Callable

import torch


def ternary_search(
    evaluate_closure: Callable[[float], torch.Tensor],
    ls_a: float = 1e-04,  # the lower bound on the step size
    ls_z: float = 1e02,  # the uppper bound on the step size
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
        fb = evaluate_closure(ls_b)
        fc = evaluate_closure(ls_c)

        # update the bounds
        if fb <= fc:  # x* must be in [a,c]
            ls_z = ls_c
            ls_optim = ls_b
        else:  # x* must be in [b,z]
            ls_a = ls_b
            ls_optim = ls_c

    assert ls_optim >= 0.0
    return ls_optim
