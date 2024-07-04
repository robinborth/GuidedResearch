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
        use_grad: bool = False,
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
        self.use_grad = use_grad

        self.step_size = step_size
        self.line_search_fn = line_search_fn

        self.step_count = 0
        self.store_system = store_system
        self.output_dir = output_dir
        self.verbose = verbose

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

    def solve_delta(self, JTJ: torch.tensor, JTF: torch.Tensor, damping_factor: float):
        """Apply the hessian approximation and solve for the delta"""
        D = torch.diag(torch.ones_like(torch.diag(JTJ)))
        if not self.levenberg:
            D = torch.diag(torch.diag(JTJ))

        # build the matrix and solve for the delta
        A = 2 * JTJ + damping_factor * D
        direction = self.lin_solver(A=A, b=JTF)

        self.save_system(A=A, x=direction, b=JTF)
        return direction

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
        loss_df = self._evaluate(loss_closure, x_init, self.step_size, dx)
        df_factor = self.damping_factor / self.factor
        dx_factor = self.solve_delta(JTJ, JTF, damping_factor=df_factor)
        loss_df_factor = self._evaluate(loss_closure, x_init, self.step_size, dx_factor)

        # both are worse -> increase damping factor until improvement
        if loss_df >= loss and loss_df_factor >= loss:
            improvement = False
            for k in range(1, self.max_damping_steps + 1):
                df_k = self.damping_factor * (self.factor**k)
                dx_k = self.solve_delta(JTJ, JTF, damping_factor=df_k)
                loss_dx_k = self._evaluate(loss_closure, x_init, self.step_size, dx_k)
                if loss_dx_k <= loss:  # improvement or same (converged)
                    improvement = True
                    break
            if not improvement and self.verbose:
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
        # loss_df = self._evaluate(loss_closure, x_init, self.step_size, dx)
        # if loss_df > loss:
        #     return torch.zeros_like(dx), loss  # ensure that the we converge
        return dx, loss

    def apply_jacobian(self, jacobian_closure: Callable[[], torch.Tensor]):
        J, F = jacobian_closure()  # (M, N)
        assert J.shape[1] == self._numel
        JTJ = J.T @ J  # (N, N)
        JTF = J.T @ F
        return JTJ, -JTF

    @torch.no_grad()
    def step(
        self,
        loss_closure: Callable[[], torch.Tensor],
        jacobian_closure: Callable[[], torch.Tensor],
    ) -> float:
        # prepare the init delta vectors
        JTJ, JTF = self.apply_jacobian(jacobian_closure)

        # difference with JTF
        with torch.enable_grad():
            self._zero_grad()
            loss = loss_closure()
            loss.backward()
            grad = self._gather_flat_grad().neg()
            if self.use_grad:
                JTF = grad  # dont use the pytorch gradients

        x_init = self._clone_param()

        # solve for the delta
        if self.mode == "static":
            direction, loss = self.static_solve(JTJ, JTF, loss_closure)
        elif self.mode == "dynamic":
            direction, loss = self.dynamic_solve(JTJ, JTF, loss_closure)
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

        return float(loss)
