import logging

import lightning as L
import torch

from lib.model.correspondence import ProjectiveCorrespondenceModule
from lib.model.flame.flame import Flame
from lib.optimizer.base import DifferentiableOptimizer
from lib.optimizer.residuals import Residuals
from lib.renderer.renderer import Renderer
from lib.tracker.logger import FlameLogger
from lib.tracker.timer import TimeTracker

log = logging.getLogger()


class OptimizerFramework(L.LightningModule):
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
            self.logger.log_step(batch, out, "train", batch_idx)
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
            self.logger.log_step(batch, out, "val", batch_idx)
        return out["loss"]

    def test_step(self, batch, batch_idx):
        out = self.model_step(batch)
        self.logger.log_step(batch, out, "val")
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


class WeightedOptimizer(OptimizerFramework):
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
            F, info = self.residuals.step(
                s_normal=batch["normal"][mask],
                s_point=batch["point"][mask],
                t_normal=out["normal"][mask],
                t_point=t_point,
                weights=weights[mask],
                params=new_params,
            )
            return F, (F, info)  # first jacobian, then two aux

        # inner optimization loop
        for optim_step in range(self.max_optims):
            self.optimizer.step(residual_closure)

        return dict(params=self.optimizer.get_params(), weights=weights)


class ICPOptimizer(OptimizerFramework):
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
        self.time_tracker = TimeTracker()
        self.verbose = True
        self.save_interval = 1

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
        datamodule = batch["datamodule"]
        converged = False
        coarse2fine = batch["coarse2fine"]
        scheduler = batch["scheduler"]

        # outer optimization loop
        self.time_tracker.start("outer_loop")
        for iter_step in range(max_iters):
            self.time_tracker.start("outer_step")
            # prepare logging
            self.reset_progress(inner_progress, max_optims)
            self.logger.iter_step = iter_step

            # build the batch
            self.time_tracker.start("fetch_data")
            coarse2fine.schedule(
                datamodule=datamodule,
                renderer=self.renderer,
                iter_step=iter_step,
            )
            batch = datamodule.fetch()
            self.time_tracker.stop("fetch_data")

            # configure the optimizer
            self.time_tracker.start("setup_optimizer")
            scheduler.configure_optimizer(
                optimizer=self.optimizer,
                iter_step=iter_step,
            )
            outer_progress.set_postfix({"params": self.optimizer._p_names})
            self.time_tracker.stop("setup_optimizer")

            # find correspondences
            self.time_tracker.start("find_correspondences")
            with torch.no_grad():
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
            self.time_tracker.stop("find_correspondences")

            # setup the residual computation
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
                F, info = self.residuals.step(
                    s_normal=batch["normal"][mask],
                    s_point=batch["point"][mask],
                    t_normal=out["normal"][mask],
                    t_point=t_point,
                    params=new_params,
                )
                return F, (F, info)

            # inner optimization loop
            self.time_tracker.start("inner_loop")
            for optim_step in range(max_optims):
                self.time_tracker.start("inner_step")

                # optimize step
                self.time_tracker.start("optimizer_step")
                self.optimizer.step(residual_closure)
                self.time_tracker.stop("optimizer_step")

                # metrics and loss logging
                self.time_tracker.start("inner_logging")
                self.logger.log_merged(
                    name="params",
                    params=self.optimizer.get_params(),
                    flame=self.flame,
                    renderer=self.renderer,
                    color=batch["color"],
                )
                loss, info = self.optimizer.loss_step(residual_closure)
                self.logger.log_metrics({"loss": loss})
                self.logger.log_metrics(info)
                inner_progress.set_postfix({"loss": loss})

                # finish the inner loop
                inner_progress.update(1)
                self.time_tracker.stop("inner_logging")
                self.time_tracker.stop("inner_step")
            self.time_tracker.stop("inner_loop")

            # progress logging
            self.time_tracker.start("outer_logging")
            if (iter_step % self.save_interval) == 0 and self.verbose:
                self.logger.log_error(
                    frame_idx=batch["frame_idx"],
                    s_point=batch["point"],
                    s_normal=batch["normal"],
                    t_mask=out["mask"],
                    t_point=out["point"],
                    t_normal=out["normal"],
                )
                self.logger.log_render(
                    frame_idx=batch["frame_idx"],
                    s_mask=batch["mask"],
                    s_point=batch["point"],
                    s_color=batch["color"],
                    t_mask=out["mask"],
                    t_point=out["point"],
                    t_color=out["color"],
                    t_normal_image=out["normal_image"],
                    t_depth_image=out["depth_image"],
                )
                self.logger.log_input_batch(
                    frame_idx=batch["frame_idx"],
                    s_mask=batch["mask"],
                    s_point=batch["point"],
                    s_normal=batch["normal"],
                    s_color=batch["color"],
                )
            self.time_tracker.stop("outer_logging")

            # finish the outer loop
            outer_progress.update(1)
            self.time_tracker.stop("outer_step")
        self.time_tracker.stop("outer_loop")

        # # final metric logging
        # self.time_tracker.print_summary()
        # self.logger.log_tracker()
        # if self.verbose:
        #     self.logger.mode = f"final/{self.mode}"
        #     self.logger.log_metrics(
        #         batch=batch,
        #         model=L.model_step(),
        #         verbose=False,
        #     )

        return dict(params=self.optimizer.get_params())


class DeepFeatureOptimizer(OptimizerFramework):
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


class VertexOptimizer(OptimizerFramework):
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
