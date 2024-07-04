import logging

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
    x0: torch.Tensor | None = None,  # dim (N)
    max_iter: int = 20,
    verbose: bool = False,
    tol: float = 1e-08,
):
    k = 0
    converged = False

    xk = torch.zeros_like(b) if x0 is None else x0  # (N)
    rk = b - A @ xk  # column vector (N)
    pk = rk

    if torch.norm(rk) < tol:
        converged = True

    while k < max_iter and not converged:
        # compute step size
        ak = (rk[None] @ rk) / (pk[None] @ A @ pk)
        # update unknowns
        xk_1 = xk + ak * pk
        # compute residuals
        rk_1 = rk - ak * A @ pk
        # compute new pk
        bk = (rk_1[None] @ rk_1) / (rk[None] @ rk)
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
        # dB = torch.linalg.solve(A, dX)
        dB = conjugate_gradient(
            A=A,
            b=dX,
            max_iter=50,
            verbose=ctx.verbose,
            tol=ctx.tol,
        )  # (N,)
        dB = torch.linalg.solve(A, dX)
        # grad_A = -grad_b * x^T
        dA = -dB[..., None] @ dX[None, ...]  # (N, N)
        return dA, dB, None, None, None


####################################################################################
# Linear System Solver + Preconditioned Conjugate Gradient
####################################################################################


class LinearSystemSolver(nn.Module):
    def forward(self, A: torch.Tensor, b: torch.Tensor):
        raise NotImplementedError()


class PytorchSolver(LinearSystemSolver):
    def forward(self, A: torch.Tensor, b: torch.Tensor):
        return torch.linalg.solve(A, b)


class PCGSolver(LinearSystemSolver):
    def __init__(
        self,
        dim: int = 1,
        max_iter: int = 20,
        verbose: bool = False,
        tol: float = 1e-08,
        mode: str = "identity",
    ):
        super().__init__()
        self.max_iter = max_iter
        self.verbose = verbose
        self.tol = tol
        if mode == "identity":
            self.condition_net: ConditionNet = IdentityConditionNet()
        elif mode == "jaccobi":
            self.condition_net = JaccobiConditionNet()
        elif mode == "fix":
            self.condition_net = FixConditionNet(dim=dim)
        elif mode == "dense":
            self.condition_net = DenseConditionNet(dim=dim)
        else:
            raise ValueError(f"The {mode=} is not supported!")

    def forward(self, A: torch.Tensor, b: torch.Tensor):
        # apply the preconditioner
        M = self.condition_net(A)  # fetch the preconditioner
        M_A = M @ A
        M_b = M @ b
        # evaluate x
        return ConjugateGradient.apply(M_A, M_b, self.max_iter, self.verbose, self.tol)


####################################################################################
# Different (Pre)-ConditionNets
####################################################################################


class ConditionNet(nn.Module):
    def forward(self, A: torch.Tensor):
        raise NotImplementedError


class IdentityConditionNet(ConditionNet):
    def forward(self, A: torch.Tensor):
        return torch.diag(torch.ones_like(torch.diag(A)))


class JaccobiConditionNet(ConditionNet):
    def forward(self, A: torch.Tensor):
        return torch.diag(1 / torch.diag(A))  # (N, N)


class FixConditionNet(ConditionNet):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.M = torch.nn.Parameter(torch.eye(dim), requires_grad=True)

    def forward(self, A: torch.Tensor):
        return self.M


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
        self.L

    def forward(self, A: torch.Tensor):
        L_flat = self.L(A.view(-1))

        # lower triangular matrix
        L = torch.zeros_like(A)
        tril_indices = torch.tril_indices(row=self.N, col=self.N, offset=0)
        L[tril_indices[0], tril_indices[1]] = L_flat

        # psd from triangular
        M = L @ L.T

        return M
