import logging
from collections import defaultdict

import lightning as L
import torch

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
        ignore = [
            "renderer",
            "flame",
            "weighting",
            "residuals",
            "regularize",
            "correspondence",
            "optimizer",
        ]
        self.save_hyperparameters(logger=False, ignore=ignore)

    def forward(self, batch: dict):
        raise NotImplementedError()

    def configure_optimizer(self):
        raise NotImplementedError()

    def model_step(self, batch: dict, mode: str):
        out = self.forward(batch)
        loss_info = self.compute_loss(batch=batch, out=out)

        # log the loss information
        for key, value in loss_info.items():
            if key.startswith("init") and self.current_epoch != 0:
                continue
            prog_bar = key == "loss"
            self.log(
                name=f"{mode}/{key}",
                value=value,
                prog_bar=prog_bar,
                on_step=True,
                on_epoch=True,
                batch_size=1,
            )

        # log optimization stats
        optim_stats = self.compute_optim_stats(out=out)
        for key, value in optim_stats.items():
            self.log(f"{mode}/{key}", value, batch_size=1)

        return dict(
            loss=loss_info["loss"],
            params=out["params"],
            weights=out["optim_weights"],
        )

    def training_step(self, batch, batch_idx):
        self.logger.mode = "train"  # type: ignore
        out = self.model_step(batch, mode="train")
        if self.perform_log_step(batch=batch, mode="train"):
            if self.current_epoch == 0:
                self.first_train_params = out["params"]
            batch["first_params"] = self.first_train_params
            self.logger.log_step(  # type: ignore
                batch=batch,
                params=out["params"],
                weights=out["weights"],
                flame=self.flame,
                renderer=self.renderer,
            )
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        self.logger.mode = "val"  # type: ignore
        out = self.model_step(batch, mode="val")
        if self.perform_log_step(batch=batch, mode="val"):
            if self.current_epoch == 0:
                self.first_val_params = out["params"]
            batch["first_params"] = self.first_val_params
            self.logger.log_step(  # type: ignore
                batch=batch,
                params=out["params"],
                weights=out["weights"],
                flame=self.flame,
                renderer=self.renderer,
            )
        return out["loss"]

    def configure_optimizers(self):
        optimizer = self.configure_optimizer()
        if self.hparams["scheduler"] is not None:
            scheduler = self.hparams["scheduler"](optimizer=optimizer)
            lr_scheduler = {"scheduler": scheduler, "monitor": self.hparams["monitor"]}
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        return {"optimizer": optimizer}

    @property
    def logger(self):
        return self._logger

    def perform_log_step(self, batch: dict, mode: str):
        return (
            batch["frame_idx"][0] == self.hparams[f"log_{mode}_frame_idx"]
            and batch["dataset"][0] == self.hparams[f"log_{mode}_dataset"]
            and (self.current_epoch % self.hparams[f"log_{mode}_interval"]) == 0
        )

    def compute_geometric_loss(
        self,
        s_mask: torch.Tensor,
        s_point: torch.Tensor,
        s_normal: torch.Tensor,
        params: dict,
    ):
        out = self.flame.render(
            renderer=self.renderer,
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
            param_loss[f"param_{p_names}"] = l1_loss * weight
        param_loss["param"] = torch.cat([p.flatten() for p in param_loss.values()])
        param_loss = {k: v.mean() for k, v in param_loss.items()}
        return param_loss

    def compute_residual_loss(self, optim_weights: list, optim_masks: list):
        loss = []
        for weight, mask in zip(optim_weights, optim_masks):
            loss.append(torch.abs(weight))
        return torch.cat(loss).mean()

    def compute_loss(self, batch: dict, out: dict):
        # extract the params
        init_params = batch["init_params"]
        gt_params = batch["params"]
        new_params = out["params"]

        # param loss
        param_loss = self.compute_param_loss(gt_params, new_params)
        init_param_loss = self.compute_param_loss(gt_params, init_params)

        # geometric loss
        geometric_loss = self.compute_geometric_loss(
            s_mask=batch["mask"],
            s_point=batch["point"],
            s_normal=batch["normal"],
            params=new_params,
        )
        init_geometric_loss = self.compute_geometric_loss(
            s_mask=batch["mask"],
            s_point=batch["point"],
            s_normal=batch["normal"],
            params=init_params,
        )
        gt_geometric_loss = self.compute_geometric_loss(
            s_mask=batch["mask"],
            s_point=batch["point"],
            s_normal=batch["normal"],
            params=gt_params,
        )

        # residual weight loss
        residual_loss = self.compute_residual_loss(
            optim_masks=out["optim_masks"],
            optim_weights=out["optim_weights"],
        )

        # final loss
        p_loss = self.hparams["param_weight"] * param_loss["param"]
        g_loss = self.hparams["geometric_weight"] * geometric_loss["point2plane"]
        r_loss = self.hparams["residual_weight"] * residual_loss
        loss = p_loss + g_loss + r_loss

        # return loss information
        return dict(
            loss=loss,
            loss_residual=r_loss,
            loss_param=p_loss,
            loss_geometric=g_loss,
            **param_loss,
            **geometric_loss,
            **{f"gt_{key}": value for key, value in gt_geometric_loss.items()},
            **{f"init_{key}": value for key, value in init_geometric_loss.items()},
            **{f"init_{key}": value for key, value in init_param_loss.items()},
        )

    def compute_optim_stats(self, out: dict):
        # extract information
        optim_weights = torch.stack(out["optim_weights"])

        # compute the optimization stats from the GN
        optim_stats = defaultdict(list)
        for optim_out in out["optim_outs"]:
            for key, value in optim_out.items():
                optim_stats[f"min_{key}"].append(value.min())
                optim_stats[f"max_{key}"].append(value.max())
                optim_stats[f"mean_{key}"].append(value.mean())
                if key == "H":
                    optim_stats[f"cond_{key}"].append(torch.linalg.cond(value))
        optim_outs = {k: torch.stack(v).mean() for k, v in optim_stats.items()}

        # compute statistics
        return dict(
            max_weight=optim_weights[-1].max(),
            min_weight=optim_weights[-1].min(),
            **optim_outs,
        )


class NeuralOptimizer(OptimizerFramework):
    def __init__(
        self,
        flame: Flame,
        correspondence: torch.nn.Module,
        weighting: torch.nn.Module,
        renderer: Renderer,
        residuals: Residuals,
        regularize: torch.nn.Module,
        optimizer: DifferentiableOptimizer,
        # logging
        logger: FlameLogger,
        verbose: bool = True,
        max_iters: int = 1,
        max_optims: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # optimizer settings
        self.flame = flame
        self.c_module = correspondence
        self.w_module = weighting
        self.r_module = regularize
        self.renderer = renderer
        self.optimizer = optimizer
        self.residuals = residuals
        self.max_iters = max_iters
        self.max_optims = max_optims
        # debugging
        self._logger = logger
        self.verbose = verbose

    def on_before_optimizer_step(self, optimizer):
        param_first = optimizer.param_groups[0]["params"][0]
        param_last = optimizer.param_groups[0]["params"][-1]
        grad = param_first.grad

        # Access moments after an update
        min_abs_grad = 100.0
        min_idx = -1
        max_abs_grad = -1.0
        max_idx = -1
        for i, param in enumerate(self.w_module.parameters()):
            state = optimizer.state[param]
            if "exp_avg" in state and "exp_avg_sq" in state:
                first_moment = state["exp_avg"]
                second_moment = state["exp_avg_sq"]
            abs_min = torch.abs(param.grad).min()
            if abs_min < min_abs_grad and abs_min > 0:
                min_abs_grad = abs_min
                min_idx = i

            abs_max = torch.abs(param.grad).max()
            if abs_max > max_abs_grad:
                max_abs_grad = abs_max
                max_idx = i

        self.log(name="debug/grad_norm", value=torch.linalg.norm(param.grad))
        self.log(name="debug/min_abs_grad", value=min_abs_grad)
        self.log(name="debug/min_idx", value=min_idx)
        self.log(name="debug/max_abs_grad", value=max_abs_grad)
        self.log(name="debug/max_idx", value=max_idx)

        if torch.isnan(grad).sum() > 0:
            print("NAN GRAD!")

    def configure_optimizer(self):
        params = [
            {"params": self.w_module.parameters()},
            {"params": self.r_module.parameters()},
        ]
        return torch.optim.Adam(params=params, lr=self.hparams["lr"])
        # return torch.optim.SGD(params=params, lr=self.hparams["lr"], momentum=0.9)

    def forward(self, batch: dict):
        optim_masks: list = []
        optim_weights: list = []
        optim_outs: list = []

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
            optim_masks.append(mask)
            # predict weights
            w_out = self.w_module(
                s_point=batch["point"],
                s_normal=batch["normal"],
                t_point=out["point"],
                t_normal=out["normal"],
            )
            optim_weights.append(w_out["weights"])
            # regress regularization
            r_out = self.r_module(
                params=self.optimizer.get_params(),
                latent=w_out["latent"],
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
                F, info = self.residuals.step(
                    s_normal=batch["normal"][mask],
                    s_point=batch["point"][mask],
                    t_normal=out["normal"][mask],
                    t_point=t_point,
                    weights=w_out["weights"][mask],
                    regularize=r_out,
                    params=new_params,
                )
                return F, (F, info)  # first jacobian, then two aux

            # inner optimization loop
            for optim_step in range(self.max_optims):
                optim_out = self.optimizer.step(residual_closure)
                optim_outs.append(optim_out)

        new_params = self.optimizer.get_params()

        return dict(
            params=new_params,
            optim_weights=optim_weights,
            optim_outs=optim_outs,
            optim_masks=optim_masks,
        )


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
        # optimizer settings
        self.flame = flame
        self.c_module = correspondence
        self.renderer = renderer
        self.optimizer = optimizer
        self.residuals = residuals
        # debuging
        self.verbose = verbose
        self.save_interval = save_interval
        self.time_tracker = TimeTracker()
        self._logger = logger

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
