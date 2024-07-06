import logging
from pathlib import Path
from typing import Any, Callable

import torch

from lib.optimizer.base import BaseOptimizer
from lib.optimizer.pcg import LinearSystemSolver

log = logging.getLogger()


class LevenbergMarquardt(BaseOptimizer):
    def __init__(
        self,
        # solver
        lin_solver: LinearSystemSolver,
        eps: float = 1e-10,
        # building the matrix A
        mode: str = "dynamic",  # dynamic, static
        max_damping_steps: int = 10,
        damping_factor: float = 1.0,
        levenberg: bool = False,
        factor: float = 2.0,
        # step size
        step_size: float = 1.0,
        line_search_fn: str | None = None,
        # store linear systems
        store_system: bool = False,
        output_dir: str = "/data",
        verbose: bool = True,
    ):
        super().__init__()

        self.mode = mode
        self.max_damping_steps = max_damping_steps
        self.init_damping_factor = damping_factor
        self.damping_factor = damping_factor
        self.factor = factor
        self.levenberg = levenberg

        self.lin_solver = lin_solver
        self.init_step_size = step_size
        self.step_size = step_size
        self.line_search_fn = line_search_fn

        self.step_count = 0
        self.store_system = store_system
        self.output_dir = output_dir
        self.verbose = verbose
        self.eps = eps

    def save_system(self, A: torch.Tensor, x: torch.Tensor, b: torch.Tensor):
        if not self.store_system:
            return
        path = Path(self.output_dir) / f"{self.step_count:07}.pt"
        path.parent.mkdir(exist_ok=True, parents=True)
        system = {"A": A.detach().cpu(), "x": x.detach().cpu(), "b": b.detach().cpu()}
        torch.save(system, path)
        self.step_count += 1

    def reset(self):
        super().reset()
        self.damping_factor = self.init_damping_factor

    def get_state(self):
        return {"damping_factor": self.damping_factor}

    def solve_delta(self, H: torch.tensor, grad_f: torch.Tensor, damping_factor: float):
        """Apply the hessian approximation and solve for the delta"""
        D = torch.diag(torch.ones_like(torch.diag(H)))
        if not self.levenberg:
            D = torch.diag(torch.diag(H))

        # build the matrix and solve for the delta
        A = H + damping_factor * D
        delta = self.lin_solver(A=A, b=grad_f)
        direction = -delta  # we need to go the negative direction

        self.save_system(A=A, x=delta, b=grad_f)
        return direction

    def dynamic_solve(
        self,
        H: torch.Tensor,
        grad_f: torch.Tensor,
        loss_closure: Callable[[], torch.Tensor],
    ):
        # prepare the init delta vectors
        x_init = self._clone_param()
        loss = loss_closure()

        dx = self.solve_delta(H, grad_f, damping_factor=self.damping_factor)
        loss_df = self._evaluate(loss_closure, x_init, self.step_size, dx)
        df_factor = self.damping_factor / self.factor
        dx_factor = self.solve_delta(H, grad_f, damping_factor=df_factor)
        loss_df_factor = self._evaluate(loss_closure, x_init, self.step_size, dx_factor)

        improvement = False
        # both are worse -> increase damping factor until improvement
        if loss_df >= loss and loss_df_factor >= loss:
            for k in range(1, self.max_damping_steps + 1):
                df_k = self.damping_factor * (self.factor**k)
                dx_k = self.solve_delta(H, grad_f, damping_factor=df_k)
                loss_dx_k = self._evaluate(loss_closure, x_init, self.step_size, dx_k)
                if loss_dx_k + self.eps < loss:
                    self.damping_factor = df_k
                    improvement = True
                    direction = dx_k
                    break
        # decrease damping factor -> more gauss newton -> bigger updates
        elif loss_df_factor + self.eps < loss:
            improvement = True
            self.damping_factor = df_factor
            direction = dx_factor
        # we improve the loss with the current damping factor no update
        elif loss_df + self.eps < loss:
            improvement = True
            direction = dx

        if not improvement:
            self.converged = True
            direction = torch.zeros_like(dx)
            if self.verbose:
                log.info(f"No improvement with {self.damping_factor=}")

        return direction, loss

    def static_solve(
        self,
        H: torch.Tensor,
        grad_f: torch.Tensor,
        loss_closure: Callable[[], torch.Tensor],
    ):
        loss = loss_closure()
        dx = self.solve_delta(H, grad_f, damping_factor=self.damping_factor)
        # loss_df = self._evaluate(loss_closure, x_init, self.step_size, dx)
        # if loss_df > loss:
        #     return torch.zeros_like(dx), loss  # ensure that the we converge
        return -dx, loss

    def apply_jacobian(self, jacobian_closure: Callable[[], torch.Tensor]):
        J, F = jacobian_closure()  # (M, N)
        assert J.shape[1] == self._numel
        H = 2 * J.T @ J  # (N, N)
        grad_f = 2 * J.T @ F
        return H, grad_f

    @torch.no_grad()
    def step(
        self,
        loss_closure: Callable[[], torch.Tensor],
        jacobian_closure: Callable[[], torch.Tensor],
    ) -> float:
        # prepare the init delta vectors
        H, grad_f = self.apply_jacobian(jacobian_closure)

        x_init = self._clone_param()

        # solve for the delta
        if self.mode == "static":
            direction, loss = self.static_solve(H, grad_f, loss_closure)
        elif self.mode == "dynamic":
            direction, loss = self.dynamic_solve(H, grad_f, loss_closure)
        else:
            ValueError(f"The mode={self.mode} is not possible.")

        # determine the step size and possible perform linesearch
        step_size = self.step_size
        if self.line_search_fn is not None:
            step_size = self.linesearch(
                loss_closure=loss_closure,
                x_init=x_init,
                direction=direction,
                line_search_fn=self.line_search_fn,
            )
        self._add_direction(step_size, direction)
        self._store_flat_grad(grad_f)

        return float(loss)
