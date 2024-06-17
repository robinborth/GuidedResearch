import torch
from torch.optim import Optimizer

from lib.optimizer.jacobian import FlameJacobian


class LevenbergMarquardt(Optimizer):
    def __init__(
        self,
        jacobian: FlameJacobian,
        params,
        damping_factor: float = 1.0,
        factor: float = 2.0,
        max_damping_steps: int = 10,
    ):
        super().__init__(params, defaults={})
        self.jacobian = jacobian
        self.damping_factor = damping_factor  # init the gamma
        self.factor = factor
        self.max_damping_steps = max_damping_steps
        self.converged = False

    def step(self):
        # init the problem
        J = self.applyJTJ()
        grad_f = self.parse_grad_f()

        # prepare the init delta vectors
        fx = self.parse_fx()
        dx = self.solve_delta(J=J, grad_f=grad_f, damping_factor=self.damping_factor)
        df_factor = self.damping_factor / self.factor
        dx_factor = self.solve_delta(J=J, grad_f=grad_f, damping_factor=df_factor)

        # compute the init loss for the trustregion
        loss = self.jacobian.loss(fx)
        loss_df = self.jacobian.loss(fx - dx)
        loss_df_factor = self.jacobian.loss(fx - dx_factor)

        # both are worse -> increase damping factor until improvement
        if loss_df >= loss and loss_df_factor >= loss:
            improvement = False
            for k in range(1, self.max_damping_steps + 1):
                df_k = self.damping_factor * (self.factor**k)
                dx_k = self.solve_delta(J=J, grad_f=grad_f, damping_factor=df_k)
                loss_dx_k = self.jacobian.loss(fx - dx_k)
                if loss_dx_k <= loss:  # improvement or same (converged)
                    improvement = True
                    break
            if not improvement:
                self.converged = True
            self.damping_factor = df_k
            self.update_params(dx_k)
        # decrease damping factor -> more gauss newton -> bigger updates
        elif loss_df_factor < loss:
            self.damping_factor = df_factor
            self.update_params(dx_factor)
        # we improve the loss with the current damping factor no update
        else:
            assert loss_df < loss
            self.update_params(dx)

    def applyJTJ(self):
        # compute the jacobian
        J = self.jacobian.build()
        assert J.shape[1] == self.jacobian.N
        return J.T @ J  # applyJTJ

    def solve_delta(self, J: torch.tensor, grad_f: torch.Tensor, damping_factor: float):
        """Apply the hessian approximation and solve for the delta"""
        H = 2 * J + damping_factor * torch.diag(torch.diag(J))
        return torch.linalg.solve(H, grad_f)

    def parse_grad_f(self):
        # build the b vector of Ax=b with is the grad
        # fill the correct spots of the gradient vector
        grad_f = torch.zeros(self.jacobian.N, device=self.jacobian.model.device)
        for group in self.param_groups:
            # extract the information and checks
            assert len(group["params"]) == 1
            param, p_name = group["params"][0], group["p_name"]

            if p_name in self.jacobian.shared_params:
                i, j = self.jacobian.offset(p_name, 0)
                grad_f[i:j] = param.grad[self.jacobian.shape_idx]
                # the gradients needs to be the same
                assert (param.grad[0] == param.grad).all()
            elif p_name in self.jacobian.unique_params:
                for b_idx, f_idx in enumerate(self.jacobian.frame_idx):
                    i, j = self.jacobian.offset(p_name, b_idx)
                    grad_f[i:j] = param.grad[f_idx]
            else:
                raise KeyError(f"No p_name with the value: {p_name}!")
        return grad_f

    def parse_fx(self):
        weight_f = torch.zeros(self.jacobian.N, device=self.jacobian.model.device)
        for group in self.param_groups:
            # extract the information and checks
            assert len(group["params"]) == 1
            param, p_name = group["params"][0], group["p_name"]

            if p_name in self.jacobian.shared_params:
                i, j = self.jacobian.offset(p_name, 0)
                weight_f[i:j] = param.data[self.jacobian.shape_idx]
                # the weights needs to be the same
                assert (param.data[0] == param.data).all()
            elif p_name in self.jacobian.unique_params:
                for b_idx, f_idx in enumerate(self.jacobian.frame_idx):
                    i, j = self.jacobian.offset(p_name, b_idx)
                    weight_f[i:j] = param.data[f_idx]
            else:
                raise KeyError(f"No p_name with the value: {p_name}!")
        return weight_f

    def update_params(self, dx: torch.Tensor):
        # update the params
        for group in self.param_groups:
            param, p_name = group["params"][0], group["p_name"]
            if p_name in self.jacobian.shared_params:
                i, j = self.jacobian.offset(p_name, 0)
                param.data[self.jacobian.shape_idx] -= dx[i:j]
            elif p_name in self.jacobian.unique_params:
                for b_idx, f_idx in enumerate(self.jacobian.frame_idx):
                    i, j = self.jacobian.offset(p_name, b_idx)
                    param.data[f_idx] -= dx[i:j]
            else:
                raise KeyError(f"No p_name with the value: {p_name}!")
