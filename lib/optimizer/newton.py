import logging
from typing import Any, Callable

import torch

from lib.optimizer.base import BaseOptimizer
from lib.optimizer.pcg import preconditioned_conjugate_gradient

log = logging.getLogger()


class LevenbergMarquardt(BaseOptimizer):
    def __init__(
        self,
        mode: str = "dynamic",  # dynamic, static
        only_levenberg: bool = False,
        max_damping_steps: int = 10,
        damping_factor: float = 1.0,
        factor: float = 2.0,
        lin_solver: str = "pytorch",  # pytorch, pcg
        pcg_steps: int = 5,
        pcg_jacobi: bool = True,
        lr: float = 1.0,
        line_search_fn: str | None = None,
    ):
        super().__init__()
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
        self.line_search_fn = line_search_fn

    @property
    def requires_jacobian(self):
        return True

    def get_state(self):
        return {"damping_factor": self.damping_factor}

    def solve_delta(self, JTJ: torch.tensor, JTF: torch.Tensor, damping_factor: float):
        """Apply the hessian approximation and solve for the delta"""
        if self.only_levenberg:
            D = damping_factor * torch.diag(torch.ones_like(torch.diag(JTJ)))
        else:
            D = damping_factor * torch.diag(torch.diag(JTJ))
        H = 2 * JTJ + D
        if self.lin_solver == "pcg":
            M = torch.diag(1 / torch.diag(H)) if self.pcg_jacobi else None  # (N, N)
            return preconditioned_conjugate_gradient(
                A=H, b=JTF, M=M, max_iter=self.pcg_steps
            )
        return torch.linalg.solve(JTJ, JTF)

    def dynamic_solve(
        self,
        JTJ: torch.Tensor,
        JTF: torch.Tensor,
        loss_closure: Callable[[], torch.Tensor],
    ):
        # prepare the init delta vectors
        x_init = self._clone_param()
        loss = loss_closure()

        dx = self.solve_delta(JTJ, JTF, damping_factor=self.damping_factor)
        loss_df = self._evaluate(loss_closure, x_init, self.lr, dx)
        df_factor = self.damping_factor / self.factor
        dx_factor = self.solve_delta(JTJ, JTF, damping_factor=df_factor)
        loss_df_factor = self._evaluate(loss_closure, x_init, self.lr, dx_factor)

        # both are worse -> increase damping factor until improvement
        if loss_df >= loss and loss_df_factor >= loss:
            improvement = False
            for k in range(1, self.max_damping_steps + 1):
                df_k = self.damping_factor * (self.factor**k)
                dx_k = self.solve_delta(JTJ, JTF, damping_factor=df_k)
                loss_dx_k = self._evaluate(loss_closure, x_init, self.lr, dx_k)
                if loss_dx_k <= loss:  # improvement or same (converged)
                    improvement = True
                    break
            if not improvement:
                log.info("No IMPROVEMENT!")
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
        JTJ: torch.Tensor,
        JTF: torch.Tensor,
        loss_closure: Callable[[], torch.Tensor],
    ):
        loss = loss_closure()
        dx = self.solve_delta(JTJ, JTF, damping_factor=self.damping_factor)
        # loss_df = self._evaluate(loss_closure, x_init, self.lr, dx)
        # if loss_df > loss:
        #     return torch.zeros_like(dx), loss  # ensure that the we converge
        return dx, loss

    def apply(self, jacobian_closure: Callable[[], torch.Tensor]):
        J, F = jacobian_closure()  # (M, N)
        assert J.shape[1] == self._numel
        JTJ = J.T @ J  # (N, N)
        JTF = J.T @ F
        return JTJ, -JTF

    @torch.no_grad()
    def newton_step(
        self,
        loss_closure: Callable[[], torch.Tensor],
        jacobian_closure: Callable[[], torch.Tensor],
    ) -> float:
        # prepare the init delta vectors
        JTJ, JTF = self.apply(jacobian_closure)

        # difference with JTF
        with torch.enable_grad():
            self._zero_grad()
            loss = loss_closure()
            loss.backward()
            grad_f = self._gather_flat_grad().neg()  # we minimize

        x_init = self._clone_param()

        # solve for the delta
        if self.mode == "static":
            direction, loss = self.static_solve(JTJ, JTF, loss_closure)
        elif self.mode == "dynamic":
            direction, loss = self.dynamic_solve(JTJ, JTF, loss_closure)
        else:
            ValueError(f"The mode={self.mode} is not possible.")

        # determine the step size and possible perform linesearch
        step_size = self.lr
        if self.line_search_fn is not None:
            step_size = self.linesearch(
                loss_closure=loss_closure,
                x_init=x_init,
                direction=direction,
                line_search_fn=self.line_search_fn,
            )
        self._add_direction(step_size, direction)

        return float(loss)
