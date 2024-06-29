import logging
from typing import Callable

import torch

from lib.optimizer.base import BaseOptimizer
from lib.optimizer.pcg import preconditioned_conjugate_gradient

log = logging.getLogger()


class LevenbergMarquardt(BaseOptimizer):
    def __init__(
        self,
        params,
        mode: str = "dynamic",  # dynamic, static
        only_levenberg: bool = False,
        damping_factor: float = 1.0,
        factor: float = 2.0,
        max_damping_steps: int = 10,
        line_search_fn: str | None = None,
        lin_solver: str = "pytorch",  # pytorch, pcg
        pcg_steps: int = 5,
        pcg_jacobi: bool = True,
        lr: float = 1.0,
    ):
        super().__init__(params, line_search_fn=line_search_fn)
        self.mode = mode
        self.damping_factor = damping_factor
        self.only_levenberg = only_levenberg
        self.factor = factor
        self.max_damping_steps = max_damping_steps
        assert lin_solver in ["pytorch", "pcg"]
        self.lin_solver = lin_solver
        self.pcg_steps = pcg_steps
        self.pcg_jacobi = pcg_jacobi
        self.lr = lr

    def get_state(self):
        return {"damping_factor": self.damping_factor}

    def applyJTJ(self, jacobian_closure: Callable[[], torch.Tensor]):
        J = jacobian_closure()  # (M, N)
        assert J.shape[1] == self._numel
        return J.T @ J  # (N, N)

    def solve_delta(self, J: torch.tensor, grad_f: torch.Tensor, damping_factor: float):
        """Apply the hessian approximation and solve for the delta"""
        if self.only_levenberg:
            D = damping_factor * torch.diag(torch.ones_like(torch.diag(J)))
        else:
            D = damping_factor * torch.diag(torch.diag(J))
        H = 2 * J + D
        if self.lin_solver == "pcg":
            M = torch.diag(1 / torch.diag(H)) if self.pcg_jacobi else None  # (N, N)
            return preconditioned_conjugate_gradient(
                A=H, b=grad_f, M=M, max_iter=self.pcg_steps
            )
        return torch.linalg.solve(H, grad_f)

    def dynamic_solve(
        self,
        loss_closure: Callable[[], torch.Tensor],
        jacobian_closure: Callable[[], torch.Tensor],
    ):
        # compute the jacobians
        J = self.applyJTJ(jacobian_closure)

        # compute the gradients
        with torch.enable_grad():
            self.zero_grad()
            loss = loss_closure()
            loss.backward()
            grad_f = self._gather_flat_grad().neg()  # we minimize

        # prepare the init delta vectors
        x_init = self._clone_param()

        dx = self.solve_delta(J=J, grad_f=grad_f, damping_factor=self.damping_factor)
        loss_df = self._evaluate(loss_closure, x_init, self.lr, dx)
        df_factor = self.damping_factor / self.factor
        dx_factor = self.solve_delta(J=J, grad_f=grad_f, damping_factor=df_factor)
        loss_df_factor = self._evaluate(loss_closure, x_init, self.lr, dx_factor)

        # both are worse -> increase damping factor until improvement
        if loss_df >= loss and loss_df_factor >= loss:
            improvement = False
            for k in range(1, self.max_damping_steps + 1):
                df_k = self.damping_factor * (self.factor**k)
                dx_k = self.solve_delta(J=J, grad_f=grad_f, damping_factor=df_k)
                loss_dx_k = self._evaluate(loss_closure, x_init, self.lr, dx_k)
                if loss_dx_k <= loss:  # improvement or same (converged)
                    improvement = True
                    break
            if not improvement:
                self.converged = True
            self.damping_factor = df_k
            return dx_k, loss

        # decrease damping factor -> more gauss newton -> bigger updates
        if loss_df_factor < loss:
            self.damping_factor = df_factor
            return dx_factor, loss

        # we improve the loss with the current damping factor no update
        assert loss_df < loss or self.line_search_fn is not None
        return dx, loss

    def static_solve(
        self,
        loss_closure: Callable[[], torch.Tensor],
        jacobian_closure: Callable[[], torch.Tensor],
    ):
        # compute the jacobians
        J = self.applyJTJ(jacobian_closure)

        # compute the gradients
        with torch.enable_grad():
            self.zero_grad()
            loss = loss_closure()
            loss.backward()
            grad_f = self._gather_flat_grad().neg()  # we minimize

        # prepare the init delta vectors
        x_init = self._clone_param()

        dx = self.solve_delta(J=J, grad_f=grad_f, damping_factor=self.damping_factor)
        loss_df = self._evaluate(loss_closure, x_init, self.lr, dx)

        if loss_df > loss:
            return torch.zeros_like(dx), loss  # ensure that the we converge
        return dx, loss

    @torch.no_grad()
    def newton_step(
        self,
        loss_closure: Callable[[], torch.Tensor],
        jacobian_closure: Callable[[], torch.Tensor],
    ) -> float:
        # prepare the init delta vectors
        x_init = self._clone_param()

        # solve for the delta
        if self.mode == "static":
            direction, loss = self.static_solve(loss_closure, jacobian_closure)
        elif self.mode == "dynamic":
            direction, loss = self.dynamic_solve(loss_closure, jacobian_closure)
        else:
            ValueError(f"The mode={self.mode} is not possible.")

        # determine the step size and possible perform linesearch
        step_size = self.lr
        if self.perform_linesearch:
            step_size = self.linesearch(
                loss_closure=loss_closure,
                x_init=x_init,
                direction=direction,
            )
        self._add_direction(step_size, direction)

        return float(loss)
