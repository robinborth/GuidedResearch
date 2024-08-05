import itertools
import logging
from typing import Any

import torch
import wandb
from tqdm import tqdm

from lib.data.datamodule import DPHMDataModule
from lib.model.flame import FLAME
from lib.model.loss import BaseLoss
from lib.model.residual_weight import ResidualWeightModule
from lib.optimizer.base import BaseOptimizer
from lib.rasterizer import Rasterizer
from lib.renderer.camera import Camera
from lib.trainer.logger import FlameLogger
from lib.trainer.scheduler import CoarseToFineScheduler, OptimizerScheduler

log = logging.getLogger()


class BaseTrainer:
    def __init__(
        self,
        model: FLAME,
        loss: BaseLoss,
        logger: FlameLogger,
        datamodule: DPHMDataModule,
        optimizer: BaseOptimizer,
        scheduler: OptimizerScheduler,
        coarse2fine: CoarseToFineScheduler,
        # camera settings
        camera: Camera,
        rasterizer: Rasterizer,
        # loop settings
        max_iters: int = 25,
        max_optims: int = 100,
        # logging settings
        save_interval: int = 1,
        final_video: bool = True,
        verbose: bool = True,
        **kwargs,
    ):
        # setup model
        self.model = model
        self.logger = logger
        self.datamodule = datamodule
        self.optimizer = optimizer
        self.loss = loss

        self.coarse2fine = coarse2fine
        self.coarse2fine.init_scheduler(camera, rasterizer)
        self.scheduler = scheduler

        # loop settings
        self.max_iters = max_iters
        self.max_optims = max_optims

        # debug settings
        self.save_interval = save_interval
        self.final_video = final_video
        self.verbose = verbose
        self.frames: list[int] = []
        self.residual_weight: Any = None

    def optimize_loop(self, outer_progress, inner_progress):
        # state
        logger = self.logger
        model = self.model
        optimizer = self.optimizer
        datamodule = self.datamodule

        # schedulers
        converged = False
        coarse2fine = self.coarse2fine
        scheduler = self.scheduler

        # outer optimization loop
        logger.time_tracker.start("outer_loop")
        for iter_step in range(self.max_iters):
            logger.time_tracker.start("outer_step")
            # prepare logging
            self.reset_progress(inner_progress, self.max_optims)
            logger.iter_step = iter_step

            # fetch single batch
            logger.time_tracker.start("fetch_batch")
            coarse2fine.schedule(
                datamodule=datamodule,
                iter_step=iter_step,
            )
            batch = datamodule.fetch()
            logger.time_tracker.stop("fetch_batch")

            # setup optimizer
            logger.time_tracker.start("setup_optimizer")
            scheduler.configure_optimizer(
                optimizer=optimizer,
                model=model,
                batch=batch,
                iter_step=iter_step,
            )
            outer_progress.set_postfix({"params": optimizer._p_names})
            logger.time_tracker.stop("setup_optimizer")

            # resets convergence when changin the energy
            energy_change = scheduler.dirty or coarse2fine.dirty
            if energy_change:
                converged = False
            if converged:
                outer_progress.update(1)
                logger.time_tracker.stop("outer_step")
                continue

            # find correspondences
            logger.time_tracker.start("find_correspondences")
            with torch.no_grad():
                correspondences = model.correspondence_step(batch)
            logger.time_tracker.stop("find_correspondences")

            # compute weight mask
            logger.time_tracker.start("compute_weight")
            if self.residual_weight is not None:
                b_depth = batch["point"][..., 2:]  # (B, H, W, 1)
                c_depth = correspondences["point"][..., 2:]  # (B, H, W, 1)
                c_normal = correspondences["normal"]  # (B, H, W, 3)
                rw_input = torch.cat([b_depth, c_depth, c_normal], dim=-1)
                rw_input = rw_input.permute(0, 3, 1, 2)  # (B, 5, H, W)
                weight_map = self.residual_weight(rw_input)
            else:
                mask = correspondences["mask"]
                weight_map = torch.ones_like(mask)
            logger.time_tracker.stop("compute_weight")

            # setup loss
            logger.time_tracker.start("setup_loss")
            L = self.loss(
                model=model,
                batch=batch,
                correspondences=correspondences,
                weight_map=weight_map,
                params=optimizer._params,
                p_names=optimizer._p_names,
                logger=logger,
            )
            loss_closure = L.loss_closure()
            jacobian_closure = L.jacobian_closure()
            logger.log_loss(L.loss_step())
            logger.time_tracker.stop("setup_loss")

            # inner optimization loop
            logger.time_tracker.start("inner_loop")
            for optim_step in range(self.max_optims):
                logger.time_tracker.start("inner_step")
                # update logger state
                logger.optim_step = optim_step
                logger.global_step = iter_step * self.max_optims + optim_step + 1
                logger.optimizer = optimizer

                # optimize step
                logger.time_tracker.start("optimizer_step")
                optimizer.step(loss_closure, jacobian_closure)
                logger.time_tracker.start("update_model", stop=True)
                scheduler.update_model(model, batch)
                logger.time_tracker.stop()

                # metrics and loss logging
                logger.time_tracker.start("inner_logging")
                if self.verbose:
                    logger.log_gradients(verbose=False)
                    logger.log_metrics(batch=batch, model=L.model_step())
                loss = logger.log_loss(L.loss_step())
                inner_progress.set_postfix({"loss": loss})

                # finish the inner loop
                inner_progress.update(1)
                logger.time_tracker.stop("inner_logging")

                if optimizer.converged:
                    converged = optim_step == 0  # outer convergence
                    logger.time_tracker.stop("inner_step")
                    break  # skip the next inner loops
                logger.time_tracker.stop("inner_step")
            logger.time_tracker.stop("inner_loop")

            # progress logging
            logger.time_tracker.start("outer_logging")
            if (iter_step % self.save_interval) == 0 and self.verbose:
                logger.log_3d_points(batch=batch, model=correspondences)
                logger.log_render(batch=batch, model=correspondences)
                logger.log_input_batch(batch=batch, model=correspondences)
                logger.log_error(batch=batch, model=correspondences)
                logger.log_params(batch=batch)
                # weight debugging
                logger.logger.log({"weight_image": wandb.Image(weight_map)})
                logger.logger.log(
                    {
                        "weight_max": weight_map.max(),
                        "weight_min": weight_map.min(),
                        "weight_mean": weight_map.mean(),
                    }
                )
                # overlay image wandb
                mask = correspondences["r_mask"]
            logger.time_tracker.stop("outer_logging")

            # finish the outer loop
            outer_progress.update(1)
            logger.time_tracker.stop("outer_step")
        logger.time_tracker.stop("outer_loop")

        # final logging of the updated correspondences
        with torch.no_grad():
            logger.iter_step += 1
            batch = datamodule.fetch()
            correspondences = model.correspondence_step(batch)
            logger.log_render(batch=batch, model=correspondences)
            batch = datamodule.fetch()
            render_merged = batch["color"].clone()
            render_merged[mask] = correspondences["color"][mask]
            img = wandb.Image(render_merged.detach().cpu().numpy())
            logger.logger.log({"merged_image": img})

        # final metric logging
        # logger.time_tracker.print_summary()
        logger.log_tracker()
        if self.verbose:
            logger.mode = f"final/{self.mode}"
            logger.log_metrics(
                batch=batch,
                model=L.model_step(),
                verbose=False,
            )

    def optimize(self):
        raise NotImplementedError

    def outer_progress(self):
        return tqdm(total=self.max_iters, desc="Outer Loop", position=1)

    def inner_progress(self):
        return tqdm(total=self.max_optims, desc="Inner Loop", leave=True, position=2)

    def close_progress(self, progresses):
        for p in progresses:
            p.close()

    def reset_progress(self, progress, total: int):
        progress.n = 0
        progress.last_print_n = 0
        progress.total = total
        progress.refresh()


class JointTrainer(BaseTrainer):
    def __init__(self, init_idxs: list[int] = [], **kwargs):
        super().__init__(**kwargs)
        self.mode = "joint"
        assert len(init_idxs) > 0
        self.init_idxs = init_idxs
        self.frames = init_idxs

    def optimize(self):
        outer_progress = self.outer_progress()
        inner_progress = self.inner_progress()

        self.model.init_params_with_config(self.model.init_config)
        self.scheduler.freeze(self.model)

        self.logger.mode = self.mode
        self.scheduler.reset()
        self.coarse2fine.reset()
        self.datamodule.update_idxs(self.init_idxs)

        self.optimize_loop(outer_progress, inner_progress)
        self.close_progress([outer_progress, inner_progress])


class SequentialTrainer(BaseTrainer):
    def __init__(
        self,
        kernel_size: int = 1,
        stride: int = 1,
        dilation: int = 1,
        start_frame: int = 0,
        end_frame: int = 126,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mode = "sequential"
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.prev_last_frame_idx = self.start_frame
        self.kernal_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        full_frames = [i for k in self.frame_idxs_iter() for i in k]
        self.frames = sorted(set(full_frames))

    def frame_progress(self):
        total = len(list(self.frame_idxs_iter()))
        return tqdm(total=total, desc="Frame Loop", position=0)

    def frame_idxs_iter(self):
        """Groups the frame idxs for optimization.
        kernel_size=3; stride=3; dilation=2
        [[0, 2, 4], [6, 8, 10]]
        """
        # defines the frame idxs to iterate over, possible with some space
        frame_idxs = list(range(self.start_frame, self.end_frame, self.dilation))
        # convoulution like iterations
        for idx in range(0, len(frame_idxs), self.stride):
            idxs = frame_idxs[idx : idx + self.kernal_size]
            if len(idxs) == self.kernal_size:
                yield idxs

    def init_frames(self, frame_idxs: list[int]):
        # initilize the kernal
        for frame_idx in frame_idxs:
            self.model.init_frame(frame_idx, self.prev_last_frame_idx)
        # set the last frame of the kernal to the one that we initlize from
        self.prev_last_frame_idx = max(frame_idxs)

    def optimize(self):
        frame_progress = self.frame_progress()
        outer_progress = self.outer_progress()
        inner_progress = self.inner_progress()
        self.model.init_params_with_config(self.model.init_config)
        self.scheduler.freeze(self.model)
        for frame_idxs in self.frame_idxs_iter():
            self.reset_progress(outer_progress, self.max_iters)
            self.logger.mode = self.mode
            self.scheduler.reset()
            self.coarse2fine.reset()
            self.datamodule.update_idxs(frame_idxs)
            self.init_frames(frame_idxs)
            self.optimize_loop(outer_progress, inner_progress)
            frame_progress.update(1)
        self.close_progress([frame_progress, outer_progress, inner_progress])


class PCGSamplingTrainer(BaseTrainer):
    def __init__(self, init_idxs: list[int] = [], max_samplings: int = 1000, **kwargs):
        super().__init__(**kwargs)
        self.mode = "pcg"
        assert len(init_idxs) > 0
        self.init_idxs = init_idxs
        self.frames = init_idxs
        self.max_samplings = max_samplings

    def sampling_progress(self):
        return tqdm(total=self.max_samplings, desc="Sampling Loop", position=0)

    def optimize(self):
        sampling_progress = self.sampling_progress()
        outer_progress = self.outer_progress()
        inner_progress = self.inner_progress()

        for _ in range(self.max_samplings):
            self.model.init_params_with_config(self.model.init_config)

            self.scheduler.freeze(self.model)
            self.logger.mode = self.mode
            self.scheduler.reset()
            self.coarse2fine.reset()
            self.datamodule.update_idxs(self.init_idxs)

            self.optimize_loop(outer_progress, inner_progress)

            sampling_progress.update(1)

        self.close_progress([sampling_progress, outer_progress, inner_progress])


class WeightingTrainer(BaseTrainer):
    def __init__(
        self,
        init_idxs: list[int] = [],
        train_steps: int = 100,
        lr: float = 1e-03,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mode = "weight"
        assert len(init_idxs) > 0
        self.init_idxs = init_idxs
        self.frames = init_idxs
        self.train_steps = train_steps

        log.info("==> initilizing weighting")
        self.residual_weight = ResidualWeightModule(
            in_channels=5,
            hidden_channels=100,
            num_layers=2,
            kernal_size=3,
        ).to(self.model.device)

        self.weight_optimizer = torch.optim.Adam(
            params=self.residual_weight.parameters(), lr=lr
        )

    def train_progress(self):
        return tqdm(total=self.train_steps, desc="Train Loop", position=0)

    def optimize(self):
        train_progress = self.train_progress()
        outer_progress = self.outer_progress()
        inner_progress = self.inner_progress()

        for train_step in range(self.train_steps):
            # setup training
            self.logger.mode = f"{self.mode}/{train_step:04}"
            self.model.init_params_with_config(self.model.init_config)
            self.scheduler.freeze(self.model)
            self.scheduler.reset()
            self.coarse2fine.reset()
            self.datamodule.update_idxs(self.init_idxs)

            # clear the gradients
            self.weight_optimizer.zero_grad()

            # perform the optimization loop
            self.optimize_loop(outer_progress, inner_progress)

            # compute the loss
            loss = []
            for params, p_names in zip(self.optimizer._params, self.optimizer._p_names):
                gt_params = self.model.init_config.get(p_names, None)
                if gt_params is not None:
                    l1_loss = torch.abs(params - torch.tensor(gt_params).to(params))
                    loss.append(l1_loss)
            loss = torch.stack(loss).mean()
            self.logger.log("optim_loss", loss)

            # perform optimization step
            loss.backward()
            self.weight_optimizer.step()

        self.close_progress([train_progress, outer_progress, inner_progress])
