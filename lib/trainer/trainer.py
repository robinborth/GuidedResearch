import itertools
import logging

import torch
from tqdm import tqdm

from lib.data.datamodule import DPHMDataModule
from lib.model.flame import FLAME
from lib.model.loss import BaseLoss
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
        # convergence tests
        check_convergence: bool = False,
        convergence_threshold: float = 1e-10,
        min_tracker_steps: int = 2,
        max_tracker_steps: int = 5,
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

        # convergence settings
        self.check_convergence = check_convergence
        self.convergence_threshold = convergence_threshold
        self.min_tracker_steps = min_tracker_steps
        self.max_tracker_steps = max_tracker_steps

        # debug settings
        self.save_interval = save_interval
        self.final_video = final_video
        self.verbose = verbose
        self.frames: list[int] = []

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
        for iter_step in range(self.max_iters):
            # prepare logging
            self.reset_progress(inner_progress, self.max_optims)
            logger.iter_step = iter_step

            # fetch single batch
            coarse2fine.schedule(
                datamodule=datamodule,
                iter_step=iter_step,
            )
            batch = datamodule.fetch()

            # setup optimizer
            scheduler.configure_optimizer(
                optimizer=optimizer,
                model=model,
                batch=batch,
                iter_step=iter_step,
            )
            outer_progress.set_postfix({"params": optimizer._p_names})

            # resets convergence when changin the energy
            energy_change = scheduler.dirty or coarse2fine.dirty
            if energy_change:
                converged = False
            if converged:
                outer_progress.update(1)
                continue

            # find correspondences
            with torch.no_grad():
                correspondences = model.correspondence_step(batch)

            # setup loss
            L = self.loss(
                model=model,
                batch=batch,
                correspondences=correspondences,
                params=optimizer._params,
                p_names=optimizer._p_names,
            )
            loss_closure = L.loss_closure()
            jacobian_closure = L.jacobian_closure()
            logger.log_loss(L.loss_step())

            # inner optimization loop
            for optim_step in range(self.max_optims):
                # update logger state
                logger.optim_step = optim_step
                logger.global_step = iter_step * self.max_optims + optim_step + 1
                logger.optimizer = optimizer

                # optimize step
                optimizer.step(loss_closure, jacobian_closure)
                scheduler.update_model(model, batch)

                # metrics and loss logging
                if self.verbose:
                    logger.log_gradients(verbose=False)
                    logger.log_metrics(batch=batch, model=L.model_step())
                loss = logger.log_loss(L.loss_step())
                inner_progress.set_postfix({"loss": loss})

                # finish the inner loop
                inner_progress.update(1)

                if optimizer.converged:
                    converged = optim_step == 0  # outer convergence
                    break  # skip the next inner loops

            # progress logging
            if (iter_step % self.save_interval) == 0 and self.verbose:
                logger.log_3d_points(batch=batch, model=correspondences)
                logger.log_render(batch=batch, model=correspondences)
                logger.log_input_batch(batch=batch, model=correspondences)
                logger.log_error(batch=batch, model=correspondences)
                logger.log_params(batch=batch)

            # finish the outer loop
            outer_progress.update(1)

        # final metric logging
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

            self.sampling_progress.update(1)

        self.close_progress([sampling_progress, outer_progress, inner_progress])
