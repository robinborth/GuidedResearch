import lightning as L
import torch

from lib.optimizer.pcg import PCGSolver


class PCGModule(L.LightningModule):
    def __init__(
        self,
        optimizer,
        dim: int = 1,
        max_iter: int = 20,
        verbose: bool = False,
        tol: float = 1e-08,
        mode: str = "identity",
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.optimizer = optimizer
        self.solver = PCGSolver(
            dim=dim,
            max_iter=max_iter,
            verbose=verbose,
            tol=tol,
            mode=mode,
            gradients="backprop",
        )

    def model_step(self, batch):
        assert len(batch["A"]) == 1
        A = batch["A"][0]
        b = batch["b"][0]

        M = self.solver.condition_net(A)
        C_m = torch.linalg.cond(M @ A).item()
        C_a = torch.linalg.cond(A).item()

        x_gt = torch.linalg.solve(A, b)
        x = self.solver(A, b)

        loss = (x - x_gt).norm() * 1000
        residual = (A @ x - b).norm()

        return dict(loss=loss, residual=residual, A_cond=C_a, M_cond=C_m)

    def training_step(self, batch, batch_idx):
        out = self.model_step(batch)
        self.log("train/loss", out["loss"], prog_bar=True)
        self.log("train/residual", out["residual"])
        self.log("train/M_cond", out["M_cond"], prog_bar=True)
        self.log("train/A_cond", out["A_cond"], prog_bar=True)
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        out = self.model_step(batch)
        self.log("val/loss", out["loss"])
        self.log("val/residual", out["residual"])
        self.log("val/M_cond", out["M_cond"])
        self.log("val/A_cond", out["A_cond"])
        return out["loss"]

    def test_step(self, batch, batch_idx):
        out = self.model_step(batch)
        self.log("test/loss", out["loss"])
        self.log("test/residual", out["residual"])
        self.log("test/M_cond", out["M_cond"])
        self.log("test/A_cond", out["A_cond"])
        return out["loss"]

    def configure_optimizers(self):
        return self.optimizer(self.solver.parameters())
