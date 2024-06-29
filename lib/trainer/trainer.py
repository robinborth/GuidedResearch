import logging

import torch
from tqdm import tqdm

from lib.data.datamodule import DPHMDataModule
from lib.model.flame import FLAME
from lib.model.loss import BaseLoss
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
        optimizer: OptimizerScheduler,
        coarse2fine: CoarseToFineScheduler,
        # camera settings
        camera: Camera,
        rasterizer: Rasterizer,
        # loop settings
        max_iters: int = 25,
        max_optims: int = 100,
        # logging settings
        save_interval: int = 1,
        # convergence tests
        check_convergence: bool = False,
        convergence_threshold: float = 1e-10,
        min_tracker_steps: int = 2,
        max_tracker_steps: int = 5,
        **kwargs,
    ):
        # setup model
        self.model = model
        self.logger = logger
        self.datamodule = datamodule
        self.loss = loss

        self.coarse2fine = coarse2fine
        self.coarse2fine.init_scheduler(camera, rasterizer)
        self.optimizer = optimizer

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
        self.frames: list[int] = []

    def optimize_loop(self, outer_progress, inner_progress):
        logger = self.logger
        model = self.model
        datamodule = self.datamodule
        coarse2fine = self.coarse2fine
        optimizer_scheduler = self.optimizer

        # outer optimization loop
        optimizer_scheduler.freeze(model)

        for iter_step in range(self.max_iters):
            # reset the inner progress bar
            self.reset_progress(inner_progress, self.max_optims)
            # prepare the logger
            logger.iter_step = iter_step

            # fetch single batch
            coarse2fine.schedule(
                datamodule=datamodule,
                iter_step=iter_step,
            )
            batch = datamodule.fetch()

            # find correspondences
            with torch.no_grad():
                correspondences = model.correspondence_step(batch)

            # setup optimizer
            optimizer = optimizer_scheduler.configure_optimizer(
                model=model,
                batch=batch,
                iter_step=iter_step,
            )

            # setup loss
            loss = self.loss(
                model=model,
                batch=batch,
                correspondences=correspondences,
                params=optimizer._params,
                p_names=optimizer._p_names,
            )
            loss_closure = loss.loss_closure()
            jacobian_closure = loss.jacobian_closure()

            # inner optimization loop
            loss_tracker = []

            for optim_step in range(self.max_optims):
                # state
                logger.optim_step = optim_step
                logger.global_step = iter_step * self.max_optims + optim_step + 1

                # optimize step
                if optimizer_scheduler.requires_jacobian:
                    loss = optimizer.newton_step(loss_closure, jacobian_closure)
                elif optimizer_scheduler.requires_loss:
                    loss = optimizer.step(loss_closure)
                else:
                    optimizer.zero_grad()
                    loss = loss_closure()
                    loss.backward()
                    optimizer.step()
                optimizer_scheduler.update_model(model, batch)

                # logging
                inner_progress.set_postfix({"loss": loss})
                logger.log(f"loss/{self.mode}", loss)
                logger.log_gradients(optimizer)

                # check for convergence
                if self.check_convergence:
                    loss_tracker.append(loss)
                    if len(loss_tracker) >= self.min_tracker_steps:
                        tracks = loss_tracker[-self.max_tracker_steps :]
                        criterion = torch.tensor(tracks).std()
                        logger.log("convergence/std", criterion)
                        logger.log(f"convergence/{iter_step:03}/std", criterion)
                        if criterion < self.convergence_threshold:
                            break

                # finish the inner loop
                inner_progress.update(1)

            # debug and logging
            logger.log_metrics(batch=batch, model=correspondences)
            if (iter_step % self.save_interval) == 0:
                logger.log_3d_points(batch=batch, model=correspondences)
                logger.log_render(batch=batch, model=correspondences)
                logger.log_input_batch(batch=batch, model=correspondences)
                logger.log_loss(batch=batch, model=correspondences)

            # finish the outer loop
            outer_progress.set_postfix({"params": optimizer._p_names, "loss": loss})
            outer_progress.update(1)

        # final metric logging
        logger.log_metrics(
            batch=batch,
            model=correspondences,
            prefix=f"metric/{self.mode}",
        )

    def optimize(self):
        raise NotImplementedError

    def frame_progress(self):
        return tqdm(total=len(self.frames), desc="Frame Loop", position=0)

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
        assert len(init_idxs) > 0
        self.init_idxs = init_idxs
        self.frames = init_idxs
        self.mode = "joint"

    def optimize(self):
        self.logger.mode = self.mode
        self.optimizer.reset()
        self.coarse2fine.reset()
        self.datamodule.update_idxs(self.init_idxs)

        outer_progress = self.outer_progress()
        inner_progress = self.inner_progress()
        self.optimize_loop(outer_progress, inner_progress)
        self.close_progress([outer_progress, inner_progress])


class SequentialTrainer(BaseTrainer):
    def __init__(self, start_frame: int = 0, end_frame: int = 126, **kwargs):
        super().__init__(**kwargs)
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.frames = list(range(self.start_frame, self.end_frame))
        self.mode = "sequential"

    def optimize(self):
        frame_progress = self.frame_progress()
        outer_progress = self.outer_progress()
        inner_progress = self.inner_progress()
        for frame_idx in range(self.start_frame, self.end_frame):
            self.reset_progress(outer_progress, self.max_iters)
            self.logger.mode = self.mode
            self.optimizer.reset()
            self.coarse2fine.reset()
            self.datamodule.update_idxs([frame_idx])

            if frame_idx != 0:
                self.model.init_frame(frame_idx)
            self.optimize_loop(outer_progress, inner_progress)
            frame_progress.update(1)
        self.close_progress([frame_progress, outer_progress, inner_progress])
