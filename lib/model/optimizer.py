import logging

import lightning as L
import matplotlib.pyplot as plt
import torch
from torch.optim.optimizer import Optimizer

import wandb
from lib.data.datamodule import DPHMDataModule
from lib.model.correspondence import ProjectiveCorrespondenceModule
from lib.model.flame.flame import Flame
from lib.optimizer.base import DifferentiableOptimizer
from lib.optimizer.residuals import Residuals
from lib.renderer.renderer import Renderer
from lib.trainer.logger import FlameLogger
from lib.trainer.scheduler import CoarseToFineScheduler, OptimizerScheduler
from lib.utils.visualize import change_color, visualize_depth_merged, visualize_grid

log = logging.getLogger()


class OptimizerModule(L.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters(
            logger=False,
            ignore=[
                "renderer",
                "flame",
                "weighting_module",
                "residuals",
                "correspondence_module",
                "optimizer",
            ],
        )

    def forward(self, batch: dict):
        raise NotImplementedError()

    def configure_optimizer(self):
        raise NotImplementedError()

    def reset_progress(self, progress, total: int):
        progress.n = 0
        progress.last_print_n = 0
        progress.total = total
        progress.refresh()

    def model_step(self, batch: dict):
        out = self.forward(batch)
        params = out["params"]
        gt_params = batch["gt_params"]
        _loss = []
        for p_names in params.keys():
            l1_loss = torch.abs(params[p_names] - gt_params[p_names])
            _loss.append(l1_loss)
        loss = torch.cat(_loss, dim=-1).mean()
        return dict(loss=loss, **out)

    def log_step(self, batch: dict, out: dict, mode: str = "train", batch_idx: int = 0):
        gt_out = self.flame.render(renderer=self.renderer, params=batch["gt_params"])
        start_out = self.flame.render(renderer=self.renderer, params=batch["params"])
        new_out = self.flame.render(renderer=self.renderer, params=out["params"])

        wandb_images = []

        images = visualize_depth_merged(
            s_color=change_color(gt_out["color"], gt_out["mask"], code=0),
            s_point=gt_out["point"],
            s_mask=gt_out["mask"],
            t_color=change_color(start_out["color"], start_out["mask"], code=1),
            t_point=start_out["point"],
            t_mask=start_out["mask"],
        )
        visualize_grid(images=images)
        wandb_images.append(wandb.Image(plt))
        plt.close()

        images = visualize_depth_merged(
            s_color=change_color(gt_out["color"], gt_out["mask"], code=0),
            s_point=gt_out["point"],
            s_mask=gt_out["mask"],
            t_color=change_color(new_out["color"], new_out["mask"], code=2),
            t_point=new_out["point"],
            t_mask=new_out["mask"],
        )
        visualize_grid(images)
        wandb_images.append(wandb.Image(plt))
        plt.close()

        images = change_color(color=gt_out["color"], mask=gt_out["mask"], code=0)
        visualize_grid(images=images)
        wandb_images.append(wandb.Image(plt))
        plt.close()

        images = change_color(color=start_out["color"], mask=start_out["mask"], code=1)
        visualize_grid(images=images)
        wandb_images.append(wandb.Image(plt))
        plt.close()

        images = change_color(color=new_out["color"], mask=new_out["mask"], code=2)
        visualize_grid(images=images)
        wandb_images.append(wandb.Image(plt))
        plt.close()

        visualize_grid(images=out["weights"])
        wandb_images.append(wandb.Image(plt))
        plt.close()

        self.logger.log_image(f"{mode}/{batch_idx}/images", wandb_images)  # type:ignore

    def training_step(self, batch, batch_idx):
        out = self.model_step(batch)
        self.log(
            "train/loss",
            out["loss"],
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            batch_size=1,
        )
        if batch_idx == 0:
            self.log_step(batch, out, "train", batch_idx)
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        out = self.model_step(batch)
        self.log(
            "val/loss",
            out["loss"],
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            batch_size=1,
        )
        if batch_idx == 0:
            self.log_step(batch, out, "val", batch_idx)
        return out["loss"]

    def test_step(self, batch, batch_idx):
        out = self.model_step(batch)
        self.log_step(batch, out, "val")
        return out["loss"]

    def configure_optimizers(self):
        optimizer = self.configure_optimizer()
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


class WeightedOptimizer(OptimizerModule):
    def __init__(
        self,
        # models
        flame: Flame,
        correspondence_module: torch.nn.Module,
        weighting_module: torch.nn.Module,
        # renderer settings
        renderer: Renderer,
        # optimization settings
        residuals: Residuals,
        optimizer: DifferentiableOptimizer,
        max_iters: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # models
        self.flame = flame
        self.c_module = correspondence_module
        self.w_module = weighting_module
        # dfferentiable renderer
        self.renderer = renderer
        # optimization settings
        self.optimizer = optimizer
        self.residuals = residuals
        self.max_iters = max_iters

    def configure_optimizer(self):
        return torch.optim.Adam(self.w_module.parameters(), lr=self.hparams["lr"])

    def forward(self, batch: dict):
        self.optimizer.set_params(batch["params"])

        # render the current state of the model
        out = self.flame.render(
            renderer=self.renderer,
            params=self.optimizer.get_params(),
        )
        # establish correspondences
        mask = self.c_module.mask(
            s_mask=batch["mask"],
            s_point=batch["point"],
            s_normal=batch["normal"],
            t_mask=out["mask"],
            t_point=out["point"],
            t_normal=out["normal"],
        )
        # predict weights
        weights = self.w_module(s_point=batch["point"], t_point=out["point"])

        def residual_closure(*args):
            # differentiable rendering without rasterization
            new_params = self.optimizer.residual_params(args)
            m_out = self.flame(**new_params)
            # recompute to perform interpolation of the point inside closure
            t_point = self.renderer.mask_interpolate(
                vertices_idx=out["vertices_idx"],
                bary_coords=out["bary_coords"],
                attributes=m_out["vertices"],
                mask=mask,
            )
            # perform the residuals
            F = self.residuals.step(
                s_normal=batch["normal"][mask],
                s_point=batch["point"][mask],
                t_normal=out["normal"][mask],
                t_point=t_point,
                weights=weights[mask],
                params=new_params,
            )
            return F, F

        # inner optimization loop
        for optim_step in range(self.max_optims):
            self.optimizer.step(residual_closure)

        return dict(params=self.optimizer.get_params(), weights=weights)


class DeepFeatureOptimizer(OptimizerModule):
    def __init__(
        self,
        # models
        flame: Flame,
        feature_module: torch.nn.Module,
        # renderer settings
        renderer: Renderer,
        # optimization settings
        residuals: Residuals,
        optimizer: DifferentiableOptimizer,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # models
        self.flame = flame
        self.c_module = ProjectiveCorrespondenceModule()
        self.f_module = feature_module
        # dfferentiable renderer
        self.renderer = renderer
        # optimization settings
        self.optimizer = optimizer
        self.residuals = residuals

    def configure_optimizer(self):
        return torch.optim.Adam(self.f_module.parameters(), lr=self.hparams["lr"])

    def forward(self, batch: dict):
        self.optimizer.set_params(batch["params"])

        # render the current state of the model
        out = self.flame.render(
            renderer=self.renderer,
            params=self.optimizer.get_params(),
        )

        # establish projective correspondences
        mask = self.c_module.mask(
            s_mask=batch["mask"],
            s_point=batch["point"],
            s_normal=batch["normal"],
            t_mask=out["mask"],
            t_point=out["point"],
            t_normal=out["normal"],
        )

        # predict features for source
        s_feature = self.f_module(batch["point"])
        t_feature = self.f_module(out["point"])

        def residual_closure(*args):
            F = self.residuals.step(
                s_feature=s_feature[mask],
                t_feature=t_feature[mask],
            )
            return F, F

        self.optimizer.step(residual_closure)
        return dict(params=self.optimizer.get_params())


class VertexOptimizer(OptimizerModule):
    def __init__(
        self,
        flame: Flame,
        optimizer: DifferentiableOptimizer,
        residuals: Residuals,
        max_iters: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.flame = flame
        self.optimizer = optimizer
        self.residuals = residuals
        self.max_iters = max_iters
        self.reqiured_params = ["vertices", "params"]

    def forward(self, batch: dict):
        self.optimizer.set_params(batch["params"])

        def residual_closure(*args):
            new_params = self.optimizer.residual_params(args)
            m_out = self.flame(**new_params)
            F = self.residuals.step(
                t_vertices=m_out["vertices"],
                s_vertices=batch["vertices"],
            )
            return F, F

        for iter_step in range(self.max_iters):
            self.optimizer.step(residual_closure)
        return dict(params=self.optimizer.get_params())


class ICPOptimizer(OptimizerModule):
    def __init__(
        self,
        flame: Flame,
        logger: FlameLogger,
        renderer: Renderer,
        correspondence: torch.nn.Module,
        residuals: Residuals,
        optimizer: DifferentiableOptimizer,
    ):
        super().__init__()
        self.flame = flame
        self.c_module = correspondence
        self.renderer = renderer
        self.optimizer = optimizer
        self.residuals = residuals
        self._logger = logger

    @property
    def logger(self):
        return self._logger

    def forward(self, batch: dict):
        self.logger.mode = batch["mode"]
        self.optimizer.set_params(batch["params"])
        outer_progress = batch["outer_progress"]
        inner_progress = batch["inner_progress"]
        max_iters = batch["max_iters"]
        max_optims = batch["max_optims"]
        coarse2fine = batch["coarse2fine"]
        scheduler = batch["scheduler"]
        datamodule = batch["datamodule"]

        for iter_step in range(max_iters):
            self.reset_progress(inner_progress, max_optims)

            # build the batch
            coarse2fine.schedule(
                datamodule=datamodule,
                renderer=self.renderer,
                iter_step=iter_step,
            )
            batch = datamodule.fetch()

            # configure the optimizer
            scheduler.configure_optimizer(
                optimizer=self.optimizer,
                iter_step=iter_step,
            )

            # render the current state of the model
            out = self.flame.render(
                renderer=self.renderer,
                params=self.optimizer.get_params(),
            )
            # establish correspondences
            mask = self.c_module.mask(
                s_mask=batch["mask"],
                s_point=batch["point"],
                s_normal=batch["normal"],
                t_mask=out["mask"],
                t_point=out["point"],
                t_normal=out["normal"],
            )

            def residual_closure(*args):
                # differentiable rendering without rasterization
                new_params = self.optimizer.residual_params(args)
                m_out = self.flame(**new_params)
                # recompute to perform interpolation of the point inside closure
                t_point = self.renderer.mask_interpolate(
                    vertices_idx=out["vertices_idx"],
                    bary_coords=out["bary_coords"],
                    attributes=m_out["vertices"],
                    mask=mask,
                )
                # perform the residuals
                F = self.residuals.step(
                    s_normal=batch["normal"][mask],
                    s_point=batch["point"][mask],
                    t_normal=out["normal"][mask],
                    t_point=t_point,
                    params=new_params,
                )
                return F, F

            # inner optimization loop
            for optim_step in range(max_optims):
                self.optimizer.step(residual_closure)
                self.logger.log_merged(
                    name="params",
                    params=self.optimizer.get_params(),
                    flame=self.flame,
                    renderer=self.renderer,
                    color=batch["color"],
                )
                inner_progress.update(1)
            outer_progress.update(1)

        return dict(params=self.optimizer.get_params())
