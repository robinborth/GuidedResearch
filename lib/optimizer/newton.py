import logging
from pathlib import Path
from typing import Any, Callable

import torch

from lib.optimizer.base import BaseOptimizer
from lib.optimizer.pcg import LinearSystemSolver

log = logging.getLogger()


class NewtonOptimizer(BaseOptimizer):
    def __init__(
        self,
        # solver
        lin_solver: LinearSystemSolver,
        # step size
        step_size: float = 1.0,
        # store linear systems
        store_system: bool = False,
        output_dir: str = "/data",
        verbose: bool = True,
    ):
        super().__init__()

        self.init_step_size = step_size
        self.step_size = step_size
        self.step_count = 0

        self.lin_solver = lin_solver

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

    def apply_jacobian(self, jacobian_closure: Callable[[], torch.Tensor]):
        J, F = jacobian_closure()  # (M, N)
        assert J.shape[1] == self._numel
        H = 2 * J.T @ J  # (N, N)
        grad_f = 2 * J.T @ F
        return J, F, H, grad_f


class GaussNewton(NewtonOptimizer):
    def __init__(
        self,
        # solver
        lin_solver: LinearSystemSolver,
        # step size
        step_size: float = 1.0,
        line_search_fn: str | None = None,
        # store linear systems
        store_system: bool = False,
        output_dir: str = "/data",
        verbose: bool = True,
    ):
        super().__init__(
            lin_solver=lin_solver,
            step_size=step_size,
            store_system=store_system,
            output_dir=output_dir,
            verbose=verbose,
        )
        self.line_search_fn = line_search_fn

    def solve_delta(self, H: torch.tensor, grad_f: torch.Tensor):
        """Apply the hessian approximation and solve for the delta"""
        delta = self.lin_solver(A=H, b=grad_f)
        direction = -delta  # we need to go the negative direction
        self.save_system(A=H, x=delta, b=grad_f)
        return direction

    @torch.no_grad()
    def step(
        self,
        loss_closure: Callable[[], torch.Tensor],
        jacobian_closure: Callable[[], torch.Tensor],
    ):
        # prepare the init delta vectors
        J, F, H, grad_f = self.apply_jacobian(jacobian_closure)

        x_init = self._clone_param()

        # solve for the direction
        loss = loss_closure()
        direction = self.solve_delta(H, grad_f)

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


class LevenbergMarquardt(NewtonOptimizer):
    def __init__(
        self,
        # solver
        lin_solver: LinearSystemSolver,
        levenberg: bool = False,
        # building the matrix A
        max_df_steps: int = 10,  # max iterations to increate the damping factor
        damping_factor: float = 1e-02,  # initial value for damping factor
        df_lower: float = 1e-07,  # lower bound for the damping factor
        df_upper: float = 1e07,  # upper bound for the damping factor
        df_up: float = 2.0,  # factor for increasing damping factor
        df_down: float = 2.0,  # factor for decreasing damping factor
        # step size
        step_size: float = 1.0,
        # convergence
        eps_step: float = 1e-10,  # determines acceptance of a LM step
        eps_grad: float = 1e-10,  # convergence tolerance for gradient
        eps_params: float = 1e-10,  # convergence tolerance for coefficients
        eps_energy: float = 1e-10,  # convergence tolerance for energy
        # store linear systems
        store_system: bool = False,
        output_dir: str = "/data",
        verbose: bool = True,
    ):
        super().__init__(
            lin_solver=lin_solver,
            step_size=step_size,
            store_system=store_system,
            output_dir=output_dir,
            verbose=verbose,
        )
        self.levenberg = levenberg

        # damping factor options
        self.max_df_steps = max_df_steps
        self.init_damping_factor = damping_factor
        self.damping_factor = damping_factor
        self.df_up = df_up
        self.df_down = df_down
        self.df_lower = df_lower
        self.df_upper = df_upper

        # convergence criterias
        self.eps_step = eps_step
        self.eps_grad = eps_grad
        self.eps_params = eps_params
        self.eps_energy = eps_energy

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

    @torch.no_grad()
    def step(
        self,
        loss_closure: Callable[[], torch.Tensor],
        jacobian_closure: Callable[[], torch.Tensor],
    ):
        # prepare the init delta vectors
        J, F, H, grad_f = self.apply_jacobian(jacobian_closure)
        M, N = J.shape

        # prepare the init delta vectors
        self._store_flat_grad(grad_f)
        x_flat = self._gather_flat_param()
        x_init = self._clone_param()
        loss_init = loss_closure()  # sum of squared residuals

        # current damping factor
        dx = self.solve_delta(H, grad_f, self.damping_factor)
        loss_df = self._evaluate(loss_closure, x_init, self.step_size, dx)

        # lower damping factor
        df_factor = max(self.damping_factor / self.df_down, self.df_lower)
        dx_factor = self.solve_delta(H, grad_f, df_factor)
        loss_df_factor = self._evaluate(loss_closure, x_init, self.step_size, dx_factor)

        # both are worse -> increase damping factor until improvement
        if loss_df >= loss_init and loss_df_factor >= loss_init:
            for _ in range(self.max_df_steps):
                df = self.damping_factor * self.df_up
                self.damping_factor = min(df, self.df_upper)
                direction = self.solve_delta(H, grad_f, self.damping_factor)
                loss = self._evaluate(loss_closure, x_init, self.step_size, direction)
                if loss < loss_init:
                    break

        # decrease damping factor -> more gauss newton -> bigger updates
        elif loss_df_factor < loss_init:
            self.damping_factor = df_factor
            direction = dx_factor
            loss = loss_df_factor

        # we improve the loss with the current damping factor no update
        elif loss_df < loss_init:
            direction = dx
            loss = loss_df

        # different convergence criterias
        if (eps := (loss_init - loss) / M) < self.eps_step:
            self.converged = True
            if self.verbose:
                log.info(f"Convergence in relative step improvement: {eps=}")
        if (eps := torch.max(torch.abs(grad_f / M))) < self.eps_grad:
            self.converged = True
            if self.verbose:
                log.info(f"Convergence in gradient: {eps=}")
        if (eps := torch.max(torch.abs(direction / x_flat))) < self.eps_params:
            self.converged = True
            if self.verbose:
                log.info(f"Convergence in params: {eps=}")
        if (eps := loss / M) < self.eps_params:
            self.converged = True
            if self.verbose:
                log.info(f"Convergence in absolute loss: {eps=}")

        if not self.converged:
            self._add_direction(self.step_size, direction)
