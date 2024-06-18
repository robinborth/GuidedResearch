# https://pytorch.org/docs/stable/_modules/torch/optim/lbfgs.html#LBFGS

from typing import Callable, Tuple

import torch
from torch.optim import Optimizer


class BaseOptimizer(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1.0,
        # max_iter=20,
        # tolerance_grad=1e-7,
        # tolerance_change=1e-9,
        line_search_fn=None,
        defaults: dict = {},
    ):
        _defaults = dict(
            lr=lr,
            # max_iter=max_iter,
            # tolerance_grad=tolerance_grad,
            # tolerance_change=tolerance_change,
            line_search_fn=line_search_fn,
            **defaults,
        )
        super().__init__(params, defaults=_defaults)

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

    def newton_step(
        self,
        loss_closure: Callable[[], torch.Tensor],
        jacobian_closure: Callable[[], torch.Tensor],
    ) -> float:
        raise NotImplementedError
