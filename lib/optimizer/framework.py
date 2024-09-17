import logging

import lightning as L
import torch

from lib.model.correspondence import ProjectiveCorrespondenceModule
from lib.model.flame.flame import Flame
from lib.optimizer.base import DifferentiableOptimizer
from lib.optimizer.residuals import LandmarkResiduals, Residuals
from lib.renderer.renderer import Renderer
from lib.tracker.logger import FlameLogger
from lib.tracker.timer import TimeTracker
from lib.utils.distance import point2plane_distance, point2point_distance
from lib.utils.progress import reset_progress

log = logging.getLogger()


class OptimizerFramework(L.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters(
            logger=False,
            ignore=[
                "renderer",
                "flame",
                "weighting",
                "residuals",
                "correspondence",
                "optimizer",
            ],
        )

    def forward(self, batch: dict):
        raise NotImplementedError()

    def configure_optimizer(self):
        raise NotImplementedError()

    def model_step(self, batch: dict, mode: str):
        # setup and inference
        self.logger.mode = mode  # type: ignore
        out = self.forward(batch)

        # extract the params
        init_params = batch["init_params"]
        gt_params = batch["params"]
        new_params = out["params"]

        # compute the loss
        loss = self.compute_param_loss(gt_params, new_params)
        self.log(
            f"{mode}/params",
            loss["loss"],
            prog_bar=False,
            on_epoch=True,
            on_step=False,
            batch_size=1,
        )
        if self.current_epoch == 0:
            self.log(
                f"{mode}/init_params",
                loss["loss"],
                prog_bar=False,
                on_epoch=True,
                on_step=False,
                batch_size=1,
            )
        for p_names in self.hparams["params"].keys():
            self.log(
                f"{mode}/params_{p_names}",
                loss[p_names].mean(),
                on_epoch=True,
                on_step=False,
                batch_size=1,
            )

        # compute the default loss
        default_loss = self.compute_param_loss(gt_params, init_params)
        self.log(
            f"{mode}/params_default",
            default_loss["loss"],
            on_epoch=True,
            on_step=False,
            batch_size=1,
        )

        # compute the metrics
        geometric_loss = self.compute_geometric_loss(
            s_mask=batch["mask"],
            s_point=batch["point"],
            s_normal=batch["normal"],
            params=new_params,
            flame=self.flame,
            renderer=self.renderer,
        )
        self.log(
            f"{mode}/point2plane",
            geometric_loss["point2plane"],
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )
        self.log(
            f"{mode}/point2point",
            geometric_loss["point2point"],
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )
        if self.current_epoch == 0:
            self.log(
                f"{mode}/init_point2point",
                geometric_loss["point2point"],
                prog_bar=False,
                on_epoch=True,
                on_step=False,
                batch_size=1,
            )

        # compute the default metrics
        default_geometric_loss = self.compute_geometric_loss(
            s_mask=batch["mask"],
            s_point=batch["point"],
            s_normal=batch["normal"],
            params=init_params,
            flame=self.flame,
            renderer=self.renderer,
        )
        self.log(
            f"{mode}/point2plane_default",
            default_geometric_loss["point2plane"],
            batch_size=1,
        )
        self.log(
            f"{mode}/point2point_default",
            default_geometric_loss["point2point"],
            batch_size=1,
        )

        gt_geometric_loss = self.compute_geometric_loss(
            s_mask=batch["mask"],
            s_point=batch["point"],
            s_normal=batch["normal"],
            params=gt_params,
            flame=self.flame,
            renderer=self.renderer,
        )
        self.log(
            f"{mode}/point2plane_gt",
            gt_geometric_loss["point2plane"],
            batch_size=1,
        )
        self.log(
            f"{mode}/point2point_gt",
            gt_geometric_loss["point2point"],
            batch_size=1,
        )

        # logging weights
        self.log(
            f"{mode}/weight_mean",
            out["weights"][-1].mean(),
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )
        self.log(
            f"{mode}/weight_max",
            out["weights"][-1].max(),
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )

        final_loss = (
            self.hparams["param_weight"] * loss["loss"]
            + self.hparams["geometric_weight"] * geometric_loss["point2plane"]
        )

        self.log(
            f"{mode}/loss",
            final_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )
        if self.current_epoch == 0:
            self.log(
                f"{mode}/init_loss",
                final_loss,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=1,
            )

        return dict(
            loss=final_loss,
            default_loss=default_loss,
            point2plane=geometric_loss["point2plane"],
            point2point=geometric_loss["point2point"],
            out=out,
            **out,
        )

    def training_step(self, batch, batch_idx):
        out = self.model_step(batch, mode="train")
        if (
            batch["frame_idx"] == self.hparams["log_frame_idx"]
            and self.current_epoch % self.hparams["log_interval"] == 0
        ):
            self.logger.log_step(  # type: ignore
                batch=batch,
                out=out["out"],
                flame=self.flame,
                renderer=self.renderer,
                mode="train",
            )
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        out = self.model_step(batch, mode="val")
        if (
            batch["frame_idx"] == 110
            and self.current_epoch % self.hparams["log_interval"] == 0
        ):
            self.logger.log_step(  # type: ignore
                batch=batch,
                out=out["out"],
                flame=self.flame,
                renderer=self.renderer,
                mode="val",
            )
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

    def compute_geometric_loss(
        self,
        s_mask: torch.Tensor,
        s_point: torch.Tensor,
        s_normal: torch.Tensor,
        params: dict,
        flame: Flame,
        renderer: Renderer,
    ):
        out = flame.render(
            renderer=renderer,
            params=params,
        )
        mask, _ = self.c_module.mask(
            s_mask=s_mask,
            s_point=s_point,
            s_normal=s_normal,
            t_mask=out["mask"],
            t_point=out["point"],
            t_normal=out["normal"],
        )
        point2plane = point2plane_distance(
            s_point[mask],
            out["point"][mask],
            out["normal"][mask],
        )
        point2plane = point2plane.mean() * 1e03  # from m to mm

        point2point = point2point_distance(
            s_point[mask],
            out["point"][mask],
        )
        point2point = point2point.mean() * 1e03  # from m to mm

        return dict(point2plane=point2plane, point2point=point2point)

    def compute_param_loss(self, params, gt_params):
        param_loss = {}
        for p_names, weight in self.hparams["params"].items():
            l1_loss = torch.abs(params[p_names] - gt_params[p_names])
            param_loss[p_names] = l1_loss * weight
        loss = torch.cat([p.flatten() for p in param_loss.values()]).mean()
        param_loss["loss"] = loss
        return param_loss


class WeightedOptimizer(OptimizerFramework):
    def __init__(
        self,
        # models
        flame: Flame,
        correspondence: torch.nn.Module,
        weighting: torch.nn.Module,
        # renderer settings
        renderer: Renderer,
        logger: FlameLogger,
        # optimization settings
        residuals: Residuals,
        optimizer: DifferentiableOptimizer,
        max_iters: int = 1,
        max_optims: int = 1,
        verbose: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # models
        self.flame = flame
        self.c_module = correspondence
        self.w_module = weighting
        # dfferentiable renderer
        self.renderer = renderer
        # optimization settings
        self.optimizer = optimizer
        self.residuals = residuals
        self.max_iters = max_iters
        self.max_optims = max_optims
        self._logger = logger

    @property
    def logger(self):
        return self._logger

    def configure_optimizer(self):
        return torch.optim.Adam(self.w_module.parameters(), lr=self.hparams["lr"])

    def forward(self, batch: dict):
        track_weights: list = []
        self.optimizer.set_params(batch["init_params"])
        self.optimizer._p_names = list(self.hparams["params"].keys())  # type: ignore

        for iter_step in range(self.max_iters):
            self.logger.iter_step = iter_step
            # render the current state of the model
            out = self.flame.render(
                renderer=self.renderer,
                params=self.optimizer.get_params(),
            )
            # establish correspondences
            mask, _ = self.c_module.mask(
                s_mask=batch["mask"],
                s_point=batch["point"],
                s_normal=batch["normal"],
                t_mask=out["mask"],
                t_point=out["point"],
                t_normal=out["normal"],
            )
            # predict weights
            weights = self.w_module(
                s_point=batch["point"],
                s_normal=batch["normal"],
                t_point=out["point"],
                t_normal=out["normal"],
            )
            track_weights.append(weights)

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

        new_params = self.optimizer.get_params()
        return dict(params=new_params, weights=track_weights)


class ICPOptimizer(OptimizerFramework):
    def __init__(
        self,
        flame: Flame,
        logger: FlameLogger,
        renderer: Renderer,
        correspondence: torch.nn.Module,
        residuals: Residuals,
        optimizer: DifferentiableOptimizer,
        save_interval: int = 1,
        verbose: bool = True,
    ):
        super().__init__()
        self.flame = flame
        self.c_module = correspondence
        self.renderer = renderer
        self.optimizer = optimizer
        self.residuals = residuals
        self._logger = logger
        self.time_tracker = TimeTracker()
        self.verbose = verbose
        self.save_interval = save_interval

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
        coarse2fine = batch["coarse2fine"]
        scheduler = batch["scheduler"]
        step_size = batch["step_size"]

        # outer optimization loop
        self.time_tracker.start("outer_loop")
        for iter_step in range(max_iters):
            self.time_tracker.start("outer_step")
            # prepare logging
            reset_progress(inner_progress, max_optims)
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
            step_size.configure_optimizer(
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
                mask, mask_info = self.c_module.mask(
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
                    s_landmark=batch["landmark"],
                    s_landmark_mask=batch["landmark_mask"],
                    t_normal=out["normal"][mask],
                    t_point=t_point,
                    t_landmark=m_out["landmark"],
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
                loss, info = self.optimizer.loss_step(residual_closure)
                inner_progress.set_postfix({"loss": loss})
                self.logger.log_loss(loss=loss, info=info)
                self.logger.log_gradients(optimizer=self.optimizer, verbose=False)

                # finish the inner loop
                inner_progress.update(1)
                self.time_tracker.stop("inner_logging")
                self.time_tracker.stop("inner_step")
            self.time_tracker.stop("inner_loop")

            # progress logging
            self.time_tracker.start("outer_logging")
            self.logger.log_live(
                frame_idx=batch["frame_idx"],
                s_color=batch["color"],
                t_mask=out["mask"],
                t_color=out["color"],
            )
            if (iter_step % self.save_interval) == 0 and self.verbose:
                self.logger.log_mask(
                    frame_idx=batch["frame_idx"],
                    masks=mask_info,
                )
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
                    t_landmark=out["landmark"],
                )
                self.logger.log_input_batch(
                    frame_idx=batch["frame_idx"],
                    s_mask=batch["mask"],
                    s_point=batch["point"],
                    s_normal=batch["normal"],
                    s_color=batch["color"],
                    s_landmark=batch["landmark"],
                )
            self.time_tracker.stop("outer_logging")

            # finish the outer loop
            outer_progress.update(1)
            self.time_tracker.stop("outer_step")
        self.time_tracker.stop("outer_loop")

        # # final state logging
        self.logger.log_tracking(
            params=self.optimizer.get_params(),
            flame=self.flame,
            renderer=self.renderer,
            s_color=batch["color"],
            s_normal=batch["normal"],
            s_mask=batch["mask"],
        )
        self.logger.log_time_tracker(self.time_tracker)
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
        mask, _ = self.c_module.mask(
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
            F, info = self.residuals.step(
                t_vertices=m_out["vertices"],
                s_vertices=batch["vertices"],
            )
            return F, (F, info)

        for iter_step in range(self.max_iters):
            self.optimizer.step(residual_closure)
        return dict(params=self.optimizer.get_params())


class LandmarkRigidOptimizer(OptimizerFramework):
    def __init__(
        self,
        flame: Flame,
        optimizer: DifferentiableOptimizer,
        max_iters: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.flame = flame
        self.optimizer = optimizer
        self.residuals = LandmarkResiduals()
        self.max_iters = max_iters

    def forward(self, batch: dict):
        self.optimizer.set_params(batch["params"])
        self.optimizer._p_names = ["transl", "global_pose"]  # type: ignore

        def residual_closure(*args):
            new_params = self.optimizer.residual_params(args)
            m_out = self.flame(**new_params)
            F, info = self.residuals.step(
                s_landmark=batch["landmark"],
                s_landmark_mask=batch["landmark_mask"],
                t_landmark=m_out["landmark"],
            )
            return F, (F, info)

        for iter_step in range(self.max_iters):
            self.optimizer.step(residual_closure)
        return dict(params=self.optimizer.get_params())
