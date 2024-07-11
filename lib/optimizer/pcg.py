import logging
from typing import Any

import lightning as L
import torch
from torch import nn

log = logging.getLogger()

####################################################################################
# Helper Functions
####################################################################################


def preconditioned_conjugate_gradient(
    A: torch.Tensor,  # dim (N,N)
    b: torch.Tensor,  # dim (N)
    x0: torch.Tensor | None = None,  # dim (N)
    max_iter: int = 20,
    verbose: bool = False,
    tol: float = 1e-08,
    M: torch.Tensor | None = None,  # dim (N,N)
):
    k = 0
    converged = False

    M = torch.diag(torch.ones_like(b)) if M is None else M  # identity
    xk = torch.zeros_like(b) if x0 is None else x0  # (N)
    rk = b - A @ xk  # column vector (N)
    zk = M @ rk  # (N)
    pk = zk

    if torch.norm(rk) < tol:
        converged = True

    while k < max_iter and not converged:
        # compute step size
        ak = (rk[None] @ zk) / (pk[None] @ A @ pk)
        # update unknowns
        xk_1 = xk + ak * pk
        # compute residuals
        rk_1 = rk - ak * A @ pk
        # compute new pk
        zk_1 = M @ rk_1
        bk = (rk_1[None] @ zk_1) / (rk[None] @ zk)
        pk_1 = zk_1 + bk * pk
        # update the next stateprint
        xk = xk_1
        pk = pk_1
        rk = rk_1
        zk = zk_1

        k += 1
        if torch.norm(rk) < tol:
            converged = True

    if verbose and converged:
        log.info(f"Converged in {k=} steps with rk={rk.norm()}.")
    if verbose and not converged:
        log.info(f"Not converged in {max_iter=} steps with rk={rk.norm()}.")

    return xk


def conjugate_gradient(
    A: torch.Tensor,  # dim (N,N)
    b: torch.Tensor,  # dim (N)
    max_iter: int = 20,
    verbose: bool = False,
    tol: float = 1e-06,
):
    # used the batched version
    if A.dim() == 3:
        return batched_conjugate_gradient(
            A=A,
            b=b,
            max_iter=max_iter,
            verbose=verbose,
            tol=tol,
        )

    k = 0
    converged = False

    xk = torch.zeros_like(b)  # (N)
    rk = b - A @ xk  # column vector (N)
    pk = rk

    if torch.norm(rk) < tol:
        converged = True

    while k < max_iter and not converged:
        # compute step size
        rk_rk = (rk * rk).sum()  # (B,)
        A_pk = A @ pk  # (B,)
        ak = rk_rk / (pk[None] @ A_pk)
        # update unknowns
        xk_1 = xk + ak * pk
        # compute residuals
        rk_1 = rk - ak * A_pk
        # compute new pk
        bk = (rk_1[None] @ rk_1) / rk_rk
        pk_1 = rk_1 + bk * pk
        # update the next stateprint
        xk = xk_1
        pk = pk_1
        rk = rk_1

        k += 1
        if torch.norm(rk) < tol:
            converged = True

    if verbose and converged:
        log.info(f"Converged in {k=} steps with rk={rk.norm()}.")
    if verbose and not converged:
        log.info(f"Not converged in {max_iter=} steps with rk={rk.norm()}.")

    return xk


def batched_conjugate_gradient(
    A: torch.Tensor,  # dim (B, N, N)
    b: torch.Tensor,  # dim (B, N)
    max_iter: int = 20,
    verbose: bool = False,
    tol: float = 1e-08,
):
    k = 0
    B = A.shape[0]
    converged = torch.zeros(B, device=b.device)
    xk = torch.zeros_like(b)  # (B, N)
    A_xk = torch.bmm(A, xk[..., None]).squeeze(-1)  # A @ xk
    rk = b - A_xk  # column vector (B, N)
    pk = rk

    # check for convergence
    residual_norm = torch.norm(rk, dim=-1)
    converged[residual_norm < tol] = 1.0

    while k < max_iter and not converged.all():
        # compute step size
        rk_rk = (rk * rk).sum(dim=-1)  # (B,)
        A_pk = torch.bmm(A, pk[..., None]).squeeze(-1)  # A @ pk (B, N)
        pk_A_pk = (pk * A_pk).sum(dim=-1)  # (B,)
        ak = (rk_rk) / (pk_A_pk)
        # update unknowns
        xk_1 = xk + ak[..., None] * pk
        # compute residuals
        rk_1 = rk - ak[..., None] * A_pk
        # compute new pk
        rk1_rk1 = (rk_1 * rk_1).sum(dim=-1)  # (B,)
        bk = rk1_rk1 / rk_rk
        pk_1 = rk_1 + bk[..., None] * pk
        # update the next stateprint
        xk = xk_1
        pk = pk_1
        rk = rk_1

        k += 1

        residual_norm = torch.norm(rk, dim=-1)
        converged[residual_norm < tol] = 1.0

    if verbose and converged.all():
        log.info(f"Converged in {k=} steps with rk={residual_norm.max()}.")
    if verbose and not converged.all():
        log.info(f"Not converged in {max_iter=} steps with rk={residual_norm.max()}.")

    return xk


class ConjugateGradient(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        A,  # dim (N,N)
        b,  # dim (N)
        max_iter=20,
        verbose=False,
        tol=1e-08,
    ):
        ctx.save_for_backward(A)
        ctx.max_iter = max_iter
        ctx.verbose = verbose
        ctx.tol = tol
        return conjugate_gradient(
            A=A,
            b=b,
            max_iter=max_iter,
            verbose=verbose,
            tol=tol,
        )

    @staticmethod
    def backward(ctx, dX):
        (A,) = ctx.saved_tensors

        # A * grad_b = grad_x
        dB = torch.linalg.solve(A, dX)  # (N,) or (B, N)

        # grad_A = -grad_b * x^T
        dA = -torch.matmul(dB.unsqueeze(-1), dX.unsqueeze(-2))

        return dA, dB, None, None, None


####################################################################################
# Linear System Solver + Preconditioned Conjugate Gradient
####################################################################################


class LinearSystemSolver(L.LightningModule):
    def forward(self, A: torch.Tensor, b: torch.Tensor):
        raise NotImplementedError()


class PytorchSolver(LinearSystemSolver):
    def forward(self, A: torch.Tensor, b: torch.Tensor):
        return torch.linalg.solve(A, b)


class PCGSolver(LinearSystemSolver):
    def __init__(
        self,
        max_iter: int = 20,
        verbose: bool = False,
        tol: float = 1e-08,
        mode: str = "identity",
        gradients: str = "close",  # close, backprop
        condition_net: Any = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.max_iter = max_iter
        self.verbose = verbose
        self.tol = tol
        assert gradients in ["close", "backprop"]
        self.gradients = gradients

        if condition_net is None:
            self.condition_net: ConditionNet = IdentityConditionNet()
        else:
            self.condition_net = condition_net

    def forward(self, A: torch.Tensor, b: torch.Tensor):
        # apply the preconditioner
        M = self.condition_net(A)  # (B, N, N) or (N, N)

        M_A = torch.matmul(M, A)  # (B, N, N) or (N, N)
        M_b = torch.matmul(M, b.unsqueeze(-1)).squeeze(-1)  # (B, N, N) or (N, N)

        # evaluate x
        if self.gradients == "close":
            return ConjugateGradient.apply(
                M_A,
                M_b,
                self.max_iter,
                False,
                self.tol,
            )
        return conjugate_gradient(
            A=M_A,
            b=M_b,
            max_iter=self.max_iter,
            verbose=False,
            tol=self.tol,
        )

    def compute_matrix_statistics(self, A: torch.Tensor, suffix: str = "A"):
        out = {}
        out["cond"] = torch.linalg.cond(A)  # (B,)
        diag = torch.linalg.diagonal(A)
        out["norm"] = torch.linalg.matrix_norm(A)
        out["diag_norm"] = torch.linalg.vector_norm(x=diag, dim=-1)
        if self.verbose:
            out["diag_min"] = diag.min(-1).values
            out["diag_max"] = diag.max(-1).values
            out["min"] = A.min(-1).values.min(-1).values
            out["max"] = A.max(-1).values.max(-1).values
        return {f"{suffix}_{k}": v for k, v in out.items()}

    def compute_residual_statistics(self, A, x, b, suffix: str = "gt"):
        out = {}
        A_x = torch.matmul(A, x.unsqueeze(-1)).squeeze(-1)
        residual = torch.abs(A_x - b)
        out["residual"] = residual.mean(dim=-1)
        out["x_norm"] = torch.linalg.vector_norm(x, dim=-1)
        out["b_norm"] = torch.linalg.vector_norm(b, dim=-1)
        if self.verbose:
            out["residual_max"] = residual.max(-1).values
            out["residual_min"] = residual.min(-1).values
        return {f"{suffix}_{k}": v for k, v in out.items()}

    def model_step(self, batch):
        A = batch["A"]
        b = batch["b"]
        x_gt = batch["x_gt"]

        M = self.condition_net(A)
        P = torch.matmul(M, A)
        x = self.forward(A, b)
        loss = torch.abs(x - x_gt)

        # statistics
        stats_A = self.compute_matrix_statistics(A, "A")
        stats_P = self.compute_matrix_statistics(P, "P")
        residual_stats_gt = self.compute_residual_statistics(A, x_gt, b, "gt")
        residual_stats_pcg = self.compute_residual_statistics(A, x, b, "pcg")

        stats_M = {}
        if self.verbose:
            stats_M = self.compute_matrix_statistics(M, "M")

        return dict(
            loss=loss,
            **stats_A,
            **stats_M,
            **stats_P,
            **residual_stats_gt,
            **residual_stats_pcg,
        )

    def log_step(self, batch: dict, out: dict, mode: str = "train"):
        self.log(f"{mode}/loss", out["loss"].mean(), prog_bar=True)
        logs = {}
        for key, value in out.items():
            if key != "loss":
                logs[f"{mode}/{key}"] = value.mean()
            if value.numel() != 1 and self.verbose:
                for idx, sys_id in enumerate(batch["sys_id"]):
                    logs[f"{mode}/{key}/{sys_id}"] = value[idx].mean()
        self.log_dict(logs)

    def training_step(self, batch, batch_idx):
        out = self.model_step(batch)
        self.log_step(batch, out, "train")
        return out["loss"].sum()

    def validation_step(self, batch, batch_idx):
        out = self.model_step(batch)
        self.log_step(batch, out, "val")
        return out["loss"].sum()

    def test_step(self, batch, batch_idx):
        out = self.model_step(batch)
        self.log_step(batch, out, "test")
        return out["loss"].sum()

    def on_before_optimizer_step(self, optimizer):
        for group in optimizer.param_groups:
            group["params"][0]

    def configure_optimizers(self):
        optimizer = self.hparams["optimizer"](self.parameters())
        if self.hparams["scheduler"] is not None:
            scheduler = self.hparams["scheduler"](optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.hparams["monitor"],
                },
            }
        return {"optimizer": optimizer}


####################################################################################
# Different (Pre)-ConditionNets
####################################################################################


class ConditionNet(L.LightningModule):
    def forward(self, A: torch.Tensor):
        """Can handle either A of dim (N, N) or the batched version (B, N, N)."""
        raise NotImplementedError


class IdentityConditionNet(ConditionNet):
    def forward(self, A: torch.Tensor):
        ones = torch.ones((A.shape[0], A.shape[1]), device=A.device)
        return torch.diag_embed(ones)


class JaccobiConditionNet(ConditionNet):
    def forward(self, A: torch.Tensor):
        diagonals = A.diagonal(dim1=-2, dim2=-1)
        return torch.diag_embed(1 / diagonals)


class FixConditionNet(ConditionNet):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.M = torch.nn.Parameter(torch.eye(dim), requires_grad=True)

    def forward(self, A: torch.Tensor):
        return self.M.expand(A.shape)


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int = 1,
        out_dim: int = 1,
        hidden_dim: int = 1,
        num_layers: int = 1,
    ):
        super().__init__()
        layers: list[Any] = []
        # input layers
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        # hidden layers
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        # output layer
        layers.append(nn.Linear(hidden_dim, out_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class DiagonalConditionNet(ConditionNet):
    def __init__(self, unknowns: int = 1, hidden_dim: int = 200, num_layers: int = 2):
        super().__init__()
        self.M = MLP(
            in_dim=unknowns * unknowns,
            out_dim=unknowns,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=1e-02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, A: torch.Tensor):
        if A.dim() == 2:
            A = A.view(-1)
        else:
            A = A.view(A.shape[0], -1)
        return torch.diag_embed(self.M(A))


class DiagonalOffsetConditionNet(ConditionNet):
    def __init__(self, unknowns: int = 1, hidden_dim: int = 200, num_layers: int = 2):
        super().__init__()
        self.N = unknowns
        self.M = MLP(
            in_dim=unknowns * unknowns,
            out_dim=unknowns,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=1e-05)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, A: torch.Tensor):
        if A.dim() == 2:
            A = A.view(-1)
        else:
            A = A.view(A.shape[0], -1)
        Identiy = torch.eye(self.N, device=A.device)
        D = torch.diag_embed(self.M(A))
        M = Identiy + D
        return M


class DenseConditionNet(ConditionNet):
    def __init__(self, dim: int = 1, diag_treshold: float = 1e-08):
        super().__init__()
        # the number of elements in the triangular matrix
        N = dim
        self.N = dim
        self.diag_threshold = diag_treshold
        tri_N = ((N * N - N) // 2) + N
        self.L = nn.Sequential(
            nn.Linear(N * N, N * N),
            nn.ReLU(),
            nn.Linear(N * N, N * N),
            nn.ReLU(),
            nn.Linear(N * N, tri_N),
        )

    def forward(self, A: torch.Tensor):
        L_flat = self.L(A.view(-1))

        # lower triangular matrix
        L = torch.zeros((self.N, self.N), device=A.device)
        tril_indices = torch.tril_indices(row=self.N, col=self.N, offset=0)
        L[tril_indices[0], tril_indices[1]] = L_flat

        # psd from triangular
        M = L @ L.T

        return M
