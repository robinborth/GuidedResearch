import logging
from pathlib import Path
from typing import Callable

import torch

from lib.optimizer.base import DifferentiableOptimizer
from lib.optimizer.solver import LinearSystemSolver

log = logging.getLogger()


class NewtonOptimizer(DifferentiableOptimizer):
    def __init__(
        self,
        # solver
        lin_solver: LinearSystemSolver,
        strategy: str = "forward-mode",
        # step size
        step_size: float = 1.0,
        # store linear systems
        store_system: bool = False,
        output_dir: str = "/data",
        verbose: bool = True,
        # convergence
        eps_step: float = 1e-10,  # determines acceptance of a optimization step
        eps_grad: float = 1e-10,  # convergence tolerance for gradient
        eps_params: float = 1e-10,  # convergence tolerance for coefficients
        eps_energy: float = 1e-10,  # convergence tolerance for energy
    ):
        super().__init__(
            verbose=verbose,
            eps_step=eps_step,
            eps_grad=eps_grad,
            eps_params=eps_params,
            eps_energy=eps_energy,
        )

        self.init_step_size = step_size
        self.step_size = step_size
        self.step_count = 0

        self.lin_solver = lin_solver
        self.strategy = strategy

        self.store_system = store_system
        self.output_dir = output_dir

    def save_system(self, A: torch.Tensor, x: torch.Tensor, b: torch.Tensor):
        if not self.store_system:
            return
        path = Path(self.output_dir) / f"{self.step_count:07}.pt"
        path.parent.mkdir(exist_ok=True, parents=True)
        x_gt = torch.linalg.solve(A, b)
        system = {
            "A": A.detach().cpu(),
            "x": x.detach().cpu(),
            "x_gt": x_gt.detach().cpu(),
            "b": b.detach().cpu(),
        }
        torch.save(system, path)
        self.step_count += 1

    def apply_jacobian(self, closure: Callable[..., torch.Tensor]):
        self.time_tracker.start("jacobian_closure")
        J, F = self.jacobian_step(closure, strategy=self.strategy)  # (M, N)
        assert J.shape[1] == self._numel
        self.time_tracker.start("H", stop=True)
        H = 2 * J.T @ J  # (N, N)
        self.time_tracker.start("grad_f", stop=True)
        grad_f = 2 * J.T @ F
        self.time_tracker.stop()
        return J, F, H, grad_f


class GaussNewton(NewtonOptimizer):
    def __init__(
        self,
        # solver
        lin_solver: LinearSystemSolver,
        strategy: str = "forward-mode",
        # step size
        step_size: float = 1.0,
        line_search_fn: str | None = None,
        # convergence
        eps_step: float = 1e-10,  # determines acceptance of a optimization step
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
            strategy=strategy,
            step_size=step_size,
            store_system=store_system,
            output_dir=output_dir,
            verbose=verbose,
            eps_step=eps_step,
            eps_grad=eps_grad,
            eps_params=eps_params,
            eps_energy=eps_energy,
        )
        self.line_search_fn = line_search_fn

    def get_state(self):
        return {}

    def solve_delta(self, H: torch.tensor, grad_f: torch.Tensor):
        """Apply the hessian approximation and solve for the delta"""
        delta, _ = self.lin_solver(A=H, b=grad_f)
        direction = -delta  # we need to go the negative direction
        self.save_system(A=H, x=delta, b=grad_f)
        return direction

    def step(self, closure: Callable[[dict[str, torch.Tensor]], torch.Tensor]):
        # prepare the init delta vectors
        self.time_tracker.start("apply_jacobian")
        J, F, H, grad_f = self.apply_jacobian(closure)

        # solve for the direction
        self.time_tracker.start("solve_delta", stop=True)
        try:
            direction = self.solve_delta(H, grad_f)
        except Exception as msg:
            log.error(f"{msg=}")
            log.error(f"{J=}")
            log.error(f"{F=}")
            log.error(f"{H=}")
            log.error(f"{grad_f=}")

        step_size = self.step_size * self._step_size_factor
        self._add_direction(step_size, direction)
        self._store_flat_grad(grad_f)
        self.time_tracker.stop()

        return dict(J=J, F=F, H=H, grad_f=grad_f, direction=direction)


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
        eps_step: float = 1e-10,  # determines acceptance of a optimization step
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
            eps_step=eps_step,
            eps_grad=eps_grad,
            eps_params=eps_params,
            eps_energy=eps_energy,
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

    def reset(self):
        super().reset()
        self.damping_factor = self.init_damping_factor

    def get_state(self):
        return {"damping_factor": self.damping_factor}

    def solve_delta(self, H: torch.tensor, grad_f: torch.Tensor, damping_factor: float):
        """Apply the hessian approximation and solve for the delta"""
        self.time_tracker.start("solve_delta")
        D = torch.diag(torch.ones_like(torch.diag(H)))
        if not self.levenberg:
            D = torch.diag(torch.diag(H))

        # build the matrix and solve for the delta
        A = H + damping_factor * D
        delta, _ = self.lin_solver(A=A, b=grad_f)
        direction = -delta  # we need to go the negative direction
        self.time_tracker.stop("solve_delta")
        self.save_system(A=A, x=delta, b=grad_f)

        return direction

    @torch.no_grad()
    def step(self, closure: Callable[[dict[str, torch.Tensor]], torch.Tensor]):
        # prepare the init delta vectors
        self.time_tracker.start("apply_jacobian")
        J, F, H, grad_f = self.apply_jacobian(closure)
        M, N = J.shape

        # prepare the init delta vectors
        self.time_tracker.start("clone_param", stop=True)
        self._store_flat_grad(grad_f)
        x_flat = self._gather_flat_param()
        x_init = self._clone_param()

        self.time_tracker.start("compute_direction", stop=True)
        loss_init, _ = self.loss_step(closure)  # sum of squared residuals

        # current damping factor
        dx = self.solve_delta(H, grad_f, self.damping_factor)
        loss_df = self.evaluate_step(closure, x_init, self.step_size, dx)

        # lower damping factor
        df_factor = max(self.damping_factor / self.df_down, self.df_lower)
        dx_factor = self.solve_delta(H, grad_f, df_factor)
        loss_df_factor = self.evaluate_step(closure, x_init, self.step_size, dx_factor)

        # both are worse -> increase damping factor until improvement
        if loss_df >= loss_init and loss_df_factor >= loss_init:
            for _ in range(self.max_df_steps):
                df = self.damping_factor * self.df_up
                self.damping_factor = min(df, self.df_upper)
                direction = self.solve_delta(H, grad_f, self.damping_factor)
                loss = self.evaluate_step(closure, x_init, self.step_size, direction)
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
        self.time_tracker.start("check_convergence", stop=True)
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
        self.time_tracker.stop()
