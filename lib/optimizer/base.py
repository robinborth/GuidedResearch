# https://pytorch.org/docs/stable/_modules/torch/optim/lbfgs.html#LBFGS

from typing import Callable, Tuple

import torch
from torch.optim import Optimizer

from lib.optimizer.linesearch import ternary_search


class BaseOptimizer(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1.0,
        line_search_fn: str | None = None,
    ):
        super().__init__(params, defaults={})

        # we only have one param group per optimization, where in params we have
        # multiple parameters which we optimize
        self._params = []
        self._p_names = []
        for group in self.param_groups:
            if len(group["params"]) != 1:
                raise ValueError("Optimizer doesn't support per-parameter options.")
            self._params.append(group["params"][0])
            self._p_names.append(group["p_name"])

        # we need to ensure that the params only contains the ones that we want to
        # optimize for, hence for all we gather gradients and could compute the jacobian
        for param in self._params:
            if not param.requires_grad:
                raise ValueError("All params in the params_group should require grads.")

        self._numel_cache = None
        self.converged = False

        # line search options
        self.lr = lr  # default lr
        self.line_search_fn = line_search_fn

    @property
    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = sum(p.numel() for p in self._params)
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, dim=0)

    def _add_direction(self, step_size, direction):
        offset = 0
        for p in self._params:
            numel = p.numel()
            p.add_(direction[offset : offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.copy_(pdata)  # inplace copy

    def _directional_evaluate(
        self,
        loss_closure: Callable[[], torch.Tensor],  # does not modify the grad
        x_init: list[torch.Tensor],
        step_size: float,
        direction: torch.Tensor,
    ) -> Tuple[float, torch.Tensor]:
        self._add_direction(step_size=step_size, direction=direction)
        self.zero_grad()
        loss = loss_closure()
        loss.backward()
        flat_grad = self._gather_flat_grad()
        self._set_param(x_init)
        return float(loss), flat_grad

    def _evaluate(
        self,
        loss_closure: Callable[[], torch.Tensor],  # does not modify the grad
        x_init: list[torch.Tensor],
        step_size: float,
        direction: torch.Tensor,
    ) -> float:
        self._add_direction(step_size=step_size, direction=direction)
        loss = loss_closure()
        self._set_param(x_init)
        return float(loss)

    def _evaluate_closure(
        self,
        loss_closure: Callable[[], torch.Tensor],
        x_init: list[torch.Tensor],
        direction: torch.Tensor,
    ):
        return lambda step_size: self._evaluate(
            loss_closure=loss_closure,
            x_init=x_init,
            step_size=step_size,
            direction=direction,
        )

    def newton_step(
        self,
        loss_closure: Callable[[], torch.Tensor],
        jacobian_closure: Callable[[], torch.Tensor],
    ) -> float:
        raise NotImplementedError

    @property
    def perform_linesearch(self):
        return self.line_search_fn is not None

    def linesearch(
        self,
        loss_closure: Callable[[], torch.Tensor],
        x_init: list[torch.Tensor],
        direction: torch.Tensor,
    ) -> float:
        if self.line_search_fn is None:
            raise ValueError("Currently no line search selected.")

        evaluate_closure = self._evaluate_closure(
            loss_closure=loss_closure,
            x_init=x_init,
            direction=direction,
        )
        if self.line_search_fn == "ternary_search":
            return ternary_search(evaluate_closure=evaluate_closure)

        raise ValueError(f"The current {self.line_search_fn=} is not supported.")
