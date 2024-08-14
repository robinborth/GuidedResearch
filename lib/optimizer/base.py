# https://pytorch.org/docs/stable/_modules/torch/optim/lbfgs.html#LBFGS

import logging
from typing import Any, Callable

import torch
from torch import nn
from torch.func import jacfwd, jacrev

from lib.trainer.timer import TimeTracker

log = logging.getLogger()


class DifferentiableOptimizer(nn.Module):
    def __init__(self):
        super().__init__()
        self._numel_cache = None
        self._params = None
        self._converged = False
        self.time_tracker = TimeTracker()

    ####################################################################################
    # Access to the FLAME parameters and State
    ####################################################################################

    def set_params(self, params: dict[str, Any]):
        self._params = {k: p.requires_grad_(True) for k, p in params.items()}

    def get_params(self):
        return self._params

    def set_state(self, state: Any):
        for key, value in state.items():
            setattr(self, key, value)

    def get_state(self):
        raise NotImplementedError

    ####################################################################################
    # Utils
    ####################################################################################

    def _reset(self):
        self._numel_cache = None
        self._params = None
        self._converged = False

    @property
    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = sum(p.numel() for p in self._params.values())
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params.values():
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, dim=0)

    def _gather_flat_param(self):
        views = []
        for p in self._params.values():
            view = p.view(-1)
            views.append(view)
        return torch.cat(views, dim=0)

    def _store_flat_grad(self, grad_f: torch.Tensor):
        offset = 0
        for p in self._params.values():
            numel = p.numel()
            p.grad = grad_f[offset : offset + numel].view_as(p)
            offset += numel
        assert offset == self._numel

    def _add_direction(self, step_size, direction):
        offset = 0
        for k, p in self._params.items():
            numel = p.numel()
            param = p + step_size * direction[offset : offset + numel].view_as(p)
            self._params[k] = param  # override the current params
            offset += numel
        assert offset == self._numel

    def _clone_param(self):
        return [
            p.clone(memory_format=torch.contiguous_format)
            for p in self._params.values()
        ]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params.values(), params_data):
            p.copy_(pdata)  # inplace copy

    def _zero_grad(self):
        for p in self._params.values():
            p.grad = None

    ####################################################################################
    # Convergence Tests
    ####################################################################################

    # @property
    # def check_convergence(self):
    #     # different convergence criterias
    #     if (eps := (loss_init - loss) / M) < self.eps_step:
    #         self.converged = True
    #         if self.verbose:
    #             log.info(f"Convergence in relative step improvement: {eps=}")
    #     if (eps := torch.max(torch.abs(grad_f / M))) < self.eps_grad:
    #         self.converged = True
    #         if self.verbose:
    #             log.info(f"Convergence in gradient: {eps=}")
    #     if (eps := torch.max(torch.abs(direction / x_flat))) < self.eps_params:
    #         self.converged = True
    #         if self.verbose:
    #             log.info(f"Convergence in params: {eps=}")
    #     if (eps := loss / M) < self.eps_params:
    #         self.converged = True
    #         if self.verbose:
    #             log.info(f"Convergence in absolute loss: {eps=}")

    ####################################################################################
    # Optimization Evaluation Steps
    ####################################################################################

    def residual_params(self, *args):
        return {k: v for k, v in zip(self._params.keys(), *args)}

    def evaluate_closure(
        self,
        closure: Callable[[dict[str, torch.Tensor]], torch.Tensor],
        x_init: list[torch.Tensor],
        direction: torch.Tensor,
    ) -> Callable[[float], float]:
        return lambda step_size: self.evaluate_step(
            closure=closure,
            x_init=x_init,
            step_size=step_size,
            direction=direction,
        )

    def evaluate_step(
        self,
        closure: Callable[[dict[str, torch.Tensor]], torch.Tensor],
        x_init: list[torch.Tensor],
        step_size: float,
        direction: torch.Tensor,
    ) -> float:
        self._add_direction(step_size=step_size, direction=direction)
        loss = self.loss_step(closure)  # not modify the grad
        self._set_param(x_init)
        return float(loss)

    def loss_step(self, closure: Callable[[dict[str, torch.Tensor]], torch.Tensor]):
        F, _ = closure(*self._params.values())
        return (F**2).sum()

    def jacobian_step(
        self,
        closure: Callable[[dict[str, torch.Tensor]], torch.Tensor],
        strategy: str = "forward-mode",
    ):
        fn = jacfwd if strategy == "forward-mode" else jacrev
        jacobian_fn = fn(
            func=closure,
            argnums=tuple(range(len(self._params))),
            has_aux=True,
        )
        jacobians, F = jacobian_fn(*self._params.values())
        J = torch.cat([j.flatten(-2) for j in jacobians], dim=-1)  # (M, N)
        return J, F

    def step(self, closure: Callable[[dict[str, torch.Tensor]], torch.Tensor]):
        """After the step the converged property is set."""
        raise NotImplementedError
