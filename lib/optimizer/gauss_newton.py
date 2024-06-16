import torch
from torch.optim import Optimizer


class GaussNewton(Optimizer):
    def __init__(self, params, batch: dict):
        super().__init__(params)
        self.shape_idx = batch["shape_idx"]
        self.frame_idx = batch["frame_idx"]

    def step(self, closure):
        J = closure()  # this closure computes the jacobian
        JTJ = J.T @ J  # applyJTJ
        xk = None
        grad_f = None
        for group in self.param_groups:
            group["params"]
        delta_x = torch.linalg.solve(JTJ, grad_f)
        xk1 = xk - delta_x
