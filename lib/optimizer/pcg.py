import logging
import time
from typing import Any

import lightning as L
import torch
from torch import nn

from lib.model.layers import MLP

log = logging.getLogger()

####################################################################################
# Helper Function
####################################################################################


def preconditioned_conjugate_gradient(
    A: torch.Tensor,  # dim (B, N, N)
    b: torch.Tensor,  # dim (B, N)
    M: torch.Tensor | None = None,  # preconditioner of dim (B, N, N)
    x0: torch.Tensor | None = None,  # dim (B, N)
    max_iter: int = 20,  # the maximum number of iterations
    rel_tol: float = 1e-06,  # tol for relative residual error ||b-Ax||/||b||
    verbose: bool = False,
    check_convergence: bool = True,  # fix, converge
):
    # stores information about the optimization
    info: dict[str, Any] = {}
    info["k"] = 0  # number of iterations
    info["converged"] = False
    info["residual_norms"] = []  # list of all residual norms /w inital residual
    info["relres_norms"] = []  # relative residual norms, /w inital relres norm
    info["xs"] = []  # list of intermediate xs that are tracked /wo inital x
    info["relres_norm"] = None  # relative residual norm, used for convergence

    # default values for unknwons and preconditioner
    b_norm = torch.linalg.vector_norm(b, dim=-1)
    if M is None:
        M = torch.diag_embed(torch.ones_like(b))  # identity
    xk = x0
    if xk is None:
        xk = torch.zeros_like(b)  # zeroes

    # handle different input dimensions
    batched = A.dim() == 3
    if not batched:
        A = A.unsqueeze(0)
        b = b.unsqueeze(0)
        M = M.unsqueeze(0)
        xk = xk.unsqueeze(0)
    B = A.shape[0]  # determines the batch dimension

    # setup optimization
    converged = torch.zeros(B, device=b.device)

    # compute initial residual
    A_xk = torch.bmm(A, xk[..., None]).squeeze(-1)  # A @ xk
    rk = b - A_xk  # column vector (B, N)

    # checks for initial convergence
    init_residual_norm = torch.linalg.vector_norm(rk, dim=-1)  # (B,)
    info["residual_norms"].append(init_residual_norm)  # tracks the residual norms
    info["relres_norm"] = init_residual_norm / b_norm
    info["relres_norms"].append(info["relres_norm"])  # convergence checks
    converged[info["relres_norm"] < rel_tol] = 1.0
    if converged.all() and check_convergence:
        xk_1 = xk  # we return xk_1
        info["converged"] = True

    # compute conjugate vector
    zk = torch.bmm(M, rk[..., None]).squeeze(-1)  # M @ rk
    pk = zk

    while info["k"] < max_iter and not info["converged"]:
        # compute step size alpha
        rk_zk = (rk * zk).sum(dim=-1)  # (B,)
        A_pk = torch.bmm(A, pk[..., None]).squeeze(-1)  # A @ pk (B, N)
        pk_A_pk = (pk * A_pk).sum(dim=-1)  # (B,)
        ak = (rk_zk) / (pk_A_pk)

        # update unknowns
        xk_1 = xk + ak[..., None] * pk
        info["xs"].append(xk_1)

        # compute residuals
        rk_1 = rk - ak[..., None] * A_pk

        # check for convergence
        residual_norm = torch.linalg.vector_norm(rk_1, dim=-1)  # (B,)
        info["residual_norms"].append(residual_norm)
        info["relres_norm"] = residual_norm / b_norm
        info["relres_norms"].append(info["relres_norm"])  # tracks the relres norms
        converged[info["relres_norm"] < rel_tol] = 1.0
        if converged.all() and check_convergence:
            info["k"] += 1
            info["converged"] = True
            break

        # compute the next conjugate vector
        zk_1 = torch.bmm(M, rk_1[..., None]).squeeze(-1)  # M @ rk_1
        rk1_zk1 = (rk_1 * zk_1).sum(dim=-1)  # (B,)
        bk = rk1_zk1 / rk_zk
        pk_1 = zk_1 + bk[..., None] * pk

        # update state
        xk = xk_1
        pk = pk_1
        rk = rk_1
        zk = zk_1
        info["k"] += 1

    # changes the dimension for non batched input
    if not batched:
        xk_1 = xk_1.squeeze(0)
        info["relres_norm"] = info["relres_norm"].squeeze(0)
        info["relres_norms"] = [n.squeeze(0) for n in info["relres_norms"]]
        info["residual_norms"] = [n.squeeze(0) for n in info["residual_norms"]]
        info["xs"] = [x.squeeze(0) for x in info["xs"]]

    # print information about the optimization
    k = info["k"]
    relres_norm = info["relres_norm"]
    if verbose and info["converged"]:
        log.info(f"Converged in {k=} steps with rk={relres_norm.max()}.")
    if verbose and not info["converged"]:
        log.info(f"Not converged in {max_iter=} steps with rk={relres_norm.max()}.")

    return xk_1, info


class ConjugateGradient(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        A,  # dim (N,N)
        b,  # dim (N)
        max_iter=20,
        verbose=False,
        rel_tol=1e-06,
        check_convergence=False,
    ):
        ctx.save_for_backward(A)
        ctx.max_iter = max_iter
        ctx.verbose = verbose
        ctx.rel_tol = rel_tol
        return preconditioned_conjugate_gradient(
            A=A,
            b=b,
            max_iter=max_iter,
            verbose=verbose,
            rel_tol=rel_tol,
            check_convergence=check_convergence,
        )

    @staticmethod
    def backward(ctx, dX, info):
        (A,) = ctx.saved_tensors

        # A * grad_b = grad_x
        dB = torch.linalg.solve(A, dX)  # (N,) or (B, N)

        # grad_A = -grad_b * x^T
        dA = -torch.matmul(dB.unsqueeze(-1), dX.unsqueeze(-2))

        return dA, dB, None, None, None, None, None


####################################################################################
# Different Losses
####################################################################################


class ResidualLoss(nn.Module):
    def forward(self, batch: dict, out: dict):
        return out["residual_norm"].mean()


class L1SolutionLoss(nn.Module):
    def forward(self, batch: dict, out: dict):
        return out["l1_solution"].mean()


class ConditionNumberLoss(nn.Module):
    def forward(self, batch: dict, out: dict):
        return out["M_cond"].mean()


class FullResidualLoss(nn.Module):
    def __init__(self, mode: str = "equal"):
        super().__init__()
        assert mode in ["equal", "none"]
        self.mode = mode

    def forward(self, batch: dict, out: dict):
        if self.mode == "eqal":
            return torch.stack(out["residual_norms"]).mean()
        # compute the residual losses for each iteration
        last_norm = out["residual_norms"][-1].mean()
        prev_norms = out["residual_norms"][1:-1]
        prev_norms = [(last_norm / n.mean()).detach() * n.mean() for n in prev_norms]
        resiudal_loss = torch.stack([last_norm, *prev_norms]).mean()
        return resiudal_loss


class SelfSupervisedLoss(nn.Module):
    def forward(self, batch: dict, out: dict):
        # compute the residual losses for each iteration
        last_norm = out["residual_norms"][-1].mean()
        prev_norms = out["residual_norms"][1:-1]
        prev_norms = [(last_norm / n.mean()).detach() * n.mean() for n in prev_norms]
        resiudal_loss = torch.stack([last_norm, *prev_norms]).mean()

        # compute the l1 solution loss for each iteration
        # x_gt = out["xs"][-1].detach()  # assume the last is the gt
        x_gt = batch["x_gt"]
        l1_solutions = [torch.abs((x - x_gt)).mean() for x in out["xs"]]
        l1_solution = torch.stack(l1_solutions).mean()

        # weight the losses
        weight = (resiudal_loss / l1_solution).detach()
        loss = (resiudal_loss + weight * l1_solution) / 2
        return loss


class ChainLoss(nn.Module):
    def __init__(
        self,
        l1_weight: float = 1.0,
        residual_weight: float = 1.0,
        cond_weight: float = 1.0,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.residual_weight = residual_weight
        self.cond_weight = cond_weight

    def forward(self, batch: dict, out: dict):
        residual_norm = out["residual_norm"].mean()
        l1_solution = out["l1_solution"].mean()
        l1_weight = (residual_norm / l1_solution).detach()
        M_cond = out["M_cond"].mean()
        cond_weight = (residual_norm / M_cond).detach()
        loss = (
            self.residual_weight * residual_norm
            + self.l1_weight * l1_solution * l1_weight
            + self.cond_weight * M_cond * cond_weight
        )
        norm_weight = self.l1_weight + self.residual_weight + self.cond_weight
        return loss / norm_weight


class WeightedSolutionResidualLoss(nn.Module):
    def __init__(self, residual_weight: float = 0.5):
        super().__init__()
        self.w = residual_weight
        assert residual_weight >= 0.0
        assert residual_weight <= 1.0

    def forward(self, batch: dict, out: dict):
        l1_solution = out["l1_solution"].mean()
        residual_norm = out["residual_norm"].mean()
        # same weight to l1 and residual
        weight = (residual_norm / l1_solution).detach()
        # convex combination of the l1 solution and the residual norm
        return (1 - self.w) * l1_solution * weight + self.w * residual_norm


class InverseLoss(nn.Module):
    def forward(self, batch: dict, out: dict):
        M = batch["M"] 
        A_inv = batch["A"].inverse()
        return torch.linalg.matrix_norm(M - A_inv).mean()


####################################################################################
# Linear System Solver + Preconditioned Conjugate Gradient
####################################################################################


class LinearSystemSolver(L.LightningModule):
    def forward(self, A: torch.Tensor, b: torch.Tensor):
        raise NotImplementedError()


class PytorchSolver(LinearSystemSolver):
    def forward(self, A: torch.Tensor, b: torch.Tensor):
        return torch.linalg.solve(A, b), {}


class PCGSolver(LinearSystemSolver):
    def __init__(
        self,
        max_iter: int = 20,
        verbose: bool = False,
        rel_tol: float = 1e-06,
        check_convergence: bool = False,
        gradients: str = "backprop",  # close, backprop
        condition_net: Any = None,
        loss: Any = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.max_iter = max_iter
        self.verbose = verbose
        self.rel_tol = rel_tol
        self.check_convergence = check_convergence

        # the gradient computation mode
        assert gradients in ["backprop", "close"]
        self.gradients = gradients
        if gradients == "close":
            log.warning(
                "Currenlty analytical gradients precompute the preconditioning system!"
            )

        self.condition_net = (
            IdentityConditionNet() if condition_net is None else condition_net()
        )
        self.loss = ResidualLoss() if loss is None else loss()

    def forward(self, A: torch.Tensor, b: torch.Tensor):
        # apply the preconditioner
        M = self.condition_net(A)  # (B, N, N) or (N, N)

        # evaluate x
        if self.gradients == "close":
            M_A = torch.matmul(M, A)  # (B, N, N) or (N, N)
            M_b = torch.matmul(M, b.unsqueeze(-1)).squeeze(-1)  # (B, N, N) or (N, N)
            x, info = ConjugateGradient.apply(
                M_A,
                M_b,
                self.max_iter,
                False,
                self.tol,
                self.check_convergence,
            )
        else:
            x, info = preconditioned_conjugate_gradient(
                A=A,
                b=b,
                M=M,
                max_iter=self.max_iter,
                verbose=False,
                rel_tol=self.rel_tol,
                check_convergence=self.check_convergence,
            )

        return x, info

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

    def compute_residual_statistics(self, A, x, b, suffix=None):
        out = {}
        A_x = torch.matmul(A, x.unsqueeze(-1)).squeeze(-1)
        out["residual_norm"] = torch.linalg.vector_norm(A_x - b, dim=-1)
        out["x_norm"] = torch.linalg.vector_norm(x, dim=-1)
        if self.verbose:
            out["residual_max"] = out["residual_norm"].max(-1).values
            out["residual_min"] = out["residual_norm"].min(-1).values
        if suffix is not None:
            return {f"{suffix}_{k}": v for k, v in out.items()}
        return out

    def model_step(self, batch):
        A = batch["A"]
        b = batch["b"]
        x_gt = batch["x_gt"]

        # compute the solution
        M = torch.matmul(self.condition_net(A), A)
        batch["M"] = M
        start_time = time.time()
        x, info = self.forward(A, b)
        measure_time = (time.time() - start_time) * 1000  # in ms

        # statistics
        l1_solution = torch.abs(x - x_gt)
        l2_solution = torch.linalg.vector_norm(x - x_gt, dim=-1)
        stats_A = self.compute_matrix_statistics(A, "A")
        stats_M = self.compute_matrix_statistics(M, "M")
        residual_stats = self.compute_residual_statistics(A, x, b)
        residual_stats_gt = self.compute_residual_statistics(A, x_gt, b, "gt")

        # combines the information
        out = dict(
            l1_solution=l1_solution,
            l2_solution=l2_solution,
            measure_time=measure_time,
            **info,
            **residual_stats,
            **residual_stats_gt,
            **stats_A,
            **stats_M,
        )

        # compute the loss
        loss = self.loss(batch, out)
        out["loss"] = loss

        return out

    def log_step(self, batch: dict, out: dict, mode: str = "train"):
        self.log(f"{mode}/loss", out["loss"], prog_bar=True)
        logs = {}
        for key, value in out.items():
            if not isinstance(value, torch.Tensor):
                continue
            if key != "loss":
                logs[f"{mode}/{key}"] = value.mean()
            if value.numel() != 1 and self.verbose:
                for idx, sys_id in enumerate(batch["sys_id"]):
                    logs[f"{mode}/{key}/{sys_id}"] = value[idx].mean()
        self.log_dict(logs)

    def training_step(self, batch, batch_idx):
        out = self.model_step(batch)
        self.log_step(batch, out, "train")
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        out = self.model_step(batch)
        self.log_step(batch, out, "val")
        return out["loss"]

    def test_step(self, batch, batch_idx):
        out = self.model_step(batch)
        self.log_step(batch, out, "val")
        return out["loss"]

    def predict_step(self, batch, batch_idx):
        out = self.model_step(batch)
        return {
            "cond": out["M_cond"],
            "relres_norms": torch.stack(out["relres_norms"]),
            "residual_norm": out["residual_norm"],
            "gt_residual_norm": out["gt_residual_norm"],
            "measure_time": torch.tensor([out["measure_time"]]),
            "iters": torch.tensor([float(out["k"])]),
        }

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
    name: str = ""

    def forward(self, A: torch.Tensor):
        """
        Can handle either A of dim (N, N) or the batched version (B, N, N).
        The condition net computes a psd matrix M that is invertible.
        """
        raise NotImplementedError


class IdentityConditionNet(ConditionNet):
    name: str = "IdentityConditionNet"

    def forward(self, A: torch.Tensor):
        ones = torch.eye(A.shape[-1], device=A.device)
        return ones.expand_as(A)


class JaccobiConditionNet(ConditionNet):
    name: str = "JaccobiConditionNet"

    def forward(self, A: torch.Tensor):
        diagonals = A.diagonal(dim1=-2, dim2=-1)
        return torch.diag_embed(1 / diagonals)


class FixConditionNet(ConditionNet):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.M = torch.nn.Parameter(torch.eye(dim), requires_grad=True)

    def forward(self, A: torch.Tensor):
        return self.M.expand_as(A)


class DiagonalConditionNet(ConditionNet):
    name: str = "DiagonalConditionNet"

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
            x = A.view(-1)
        else:
            x = A.view(A.shape[0], -1)
        D = self.M(x)
        Identiy = torch.ones_like(D)
        M = torch.diag_embed((D + Identiy) ** 2)
        return M


class DiagonalSigmoidConditionNet(ConditionNet):
    name: str = "DiagonalSigmoidConditionNet"

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
            x = A.view(-1)
        else:
            x = A.view(A.shape[0], -1)
        D = self.M(x)
        Identiy = torch.ones_like(D)
        M = torch.diag_embed(torch.nn.functional.sigmoid((D + Identiy)))
        return M


class DenseConditionNet(ConditionNet):
    name: str = "DenseConditionNet"

    def __init__(self, unknowns: int = 1, hidden_dim: int = 200, num_layers: int = 2):
        super().__init__()
        # the number of elements in the triangular matrix
        self.N = unknowns
        self.M = MLP(
            in_dim=unknowns * unknowns,
            out_dim=unknowns * unknowns,
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
            x = A.view(-1)
        else:
            x = A.view(A.shape[0], -1)
        D = self.M(x).reshape(A.shape)
        Identiy = torch.eye(A.shape[-1], device=A.device).expand_as(A)
        M = D + Identiy
        return torch.matmul(M.transpose(-2, -1), M)  # M.T @ M to make PSD
