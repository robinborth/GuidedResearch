import logging

import torch

log = logging.getLogger()


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


class ConjugateGradient(torch.autograd.Function):
    def __init__(
        self,
        A: torch.Tensor,  # dim (N,N)
        b: torch.Tensor,  # dim (N)
        x0: torch.Tensor | None = None,  # dim (N)
        max_iter: int = 20,
        verbose: bool = False,
        tol: float = 1e-08,
        M: torch.Tensor | None = None,  # dim (N,N)
    ):
        self.tol = tol

    @staticmethod
    def forward(ctx, A: torch.Tensor, b: torch.Tensor):
        return preconditioned_conjugate_gradient(A, b)
