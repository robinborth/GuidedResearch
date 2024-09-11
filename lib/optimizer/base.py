# https://pytorch.org/docs/stable/_modules/torch/optim/lbfgs.html#LBFGS

import logging
from typing import Any, Callable

import torch
from torch.func import jacfwd, jacrev

from lib.tracker.timer import TimeTracker

log = logging.getLogger()


class DifferentiableOptimizer:
    def __init__(
        self,
        verbose: bool = True,
        eps_step: float = 1e-10,  # determines acceptance of a optimization step
        eps_grad: float = 1e-10,  # convergence tolerance for gradient
        eps_params: float = 1e-10,  # convergence tolerance for coefficients
        eps_energy: float = 1e-10,  # convergence tolerance for energy
    ):
        self._params = None
        self._p_names = None
        self._converged = False
        self.time_tracker = TimeTracker()
        self.residual_tracker = []

        # convergence criterias
        self.eps_step = eps_step
        self.eps_grad = eps_grad
        self.eps_params = eps_params
        self.eps_energy = eps_energy
        self.verbose = verbose

    ####################################################################################
    # Access to the FLAME parameters and State
    ####################################################################################

    def set_params(self, params: dict[str, Any]):
        self._p_names = list(params.keys())
        self._params = {k: p.requires_grad_(True) for k, p in params.items()}

    @property
    def _aktive_params(self):
        return {k: v for k, v in self._params.items() if k in self._p_names}

    @property
    def _default_params(self):
        return {k: v for k, v in self._params.items() if k not in self._p_names}

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
        self._params = None
        self._p_names = None
        self._converged = False

    @property
    def _numel(self):
        return sum(p.numel() for p in self._aktive_params.values())

    def _gather_flat_grad(self):
        views = []
        for p in self._aktive_params.values():
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, dim=0)

    def _gather_flat_param(self):
        views = []
        for p in self._aktive_params.values():
            view = p.view(-1)
            views.append(view)
        return torch.cat(views, dim=0)

    def _store_flat_grad(self, grad_f: torch.Tensor):
        offset = 0
        for p in self._aktive_params.values():
            numel = p.numel()
            p.grad = grad_f[offset : offset + numel].view_as(p)
            offset += numel
        assert offset == self._numel

    def _add_direction(self, step_size, direction):
        offset = 0
        for k, p in self._aktive_params.items():
            numel = p.numel()
            param = p + step_size * direction[offset : offset + numel].view_as(p)
            self._params[k] = param  # override the current params
            offset += numel
        assert offset == self._numel

    def _clone_param(self):
        return [
            p.clone(memory_format=torch.contiguous_format)
            for p in self._aktive_params.values()
        ]

    def _set_param(self, params_data):
        for p, pdata in zip(self._aktive_params.values(), params_data):
            p.copy_(pdata)  # inplace copy

    def _zero_grad(self):
        for p in self._aktive_params.values():
            p.grad = None

    ####################################################################################
    # Convergence Tests
    ####################################################################################

    # def check_convergence(
    #     self,
    #     loss_init: torch.Tensor,
    #     loss: torch.Tensor,
    #     grad_f: torch.Tensor,
    #     x_init: torch.Tensor,
    #     x: torch.Tensor,
    # ):
    #     # different convergence criterias
    #     if (eps := (loss_init - loss)) < self.eps_step:
    #         self.converged = True
    #         if self.verbose:
    #             log.info(f"Convergence in relative step improvement: {eps=}")
    #     if (eps := torch.max(torch.abs(grad_f))) < self.eps_grad:
    #         self.converged = True
    #         if self.verbose:
    #             log.info(f"Convergence in gradient: {eps=}")
    #     if (eps := torch.max(torch.abs(direction))) < self.eps_params:
    #         self.converged = True
    #         if self.verbose:
    #             log.info(f"Convergence in params: {eps=}")
    #     if (eps := loss) < self.eps_params:
    #         self.converged = True
    #         if self.verbose:
    #             log.info(f"Convergence in absolute loss: {eps=}")

    ####################################################################################
    # Optimization Evaluation Steps
    ####################################################################################

    def residual_params(self, *args):
        out = {}
        for p_name, param in zip(self._aktive_params.keys(), *args):
            out[p_name] = param
        for p_name, param in self._default_params.items():
            out[p_name] = param
        return out

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
        loss, _ = self.loss_step(closure)  # not modify the grad
        self._set_param(x_init)
        return float(loss)

    def loss_step(self, closure: Callable[[dict[str, torch.Tensor]], torch.Tensor]):
        F, (_, info) = closure(*self._aktive_params.values())
        loss = (F**2).sum()
        info = {k: (r**2).sum() for k, r in info.items()}
        return loss, info

    def jacobian_step(
        self,
        closure: Callable[[dict[str, torch.Tensor]], torch.Tensor],
        strategy: str = "forward-mode",
    ):
        fn = jacfwd if strategy == "forward-mode" else jacrev
        jacobian_fn = fn(
            func=closure,
            argnums=tuple(range(len(self._aktive_params))),
            has_aux=True,
        )
        jacobians, (F, info) = jacobian_fn(*self._aktive_params.values())
        J = torch.cat([j.flatten(-2) for j in jacobians], dim=-1)  # (M, N)
        self.residual_tracker.append(int(J.shape[0]))
        return J, F

    def step(self, closure: Callable[[dict[str, torch.Tensor]], torch.Tensor]):
        """After the step the converged property is set."""
        raise NotImplementedError
