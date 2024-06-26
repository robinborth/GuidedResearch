import logging

import torch
from tqdm import tqdm

from lib.data.datamodule import DPHMDataModule
from lib.model.flame import FLAME
from lib.rasterizer import Rasterizer
from lib.renderer.camera import Camera
from lib.trainer.logger import FlameLogger
from lib.trainer.scheduler import CoarseToFineScheduler, OptimizerScheduler

log = logging.getLogger()


class BaseTrainer:
    def __init__(
        self,
        model: FLAME,
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

    def optimize_loop(self):
        logger = self.logger
        model = self.model
        datamodule = self.datamodule
        coarse2fine = self.coarse2fine
        optimizer_scheduler = self.optimizer

        # outer optimization loop
        optimizer_scheduler.freeze(model)
        for iter_step in tqdm(range(self.max_iters)):
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

            # closures
            loss_closure = model.loss_closure(batch, correspondences, optimizer)
            jacobian_closure = model.jacobian_closure(batch, correspondences, optimizer)

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
                logger.log("loss/final", loss)
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

            # debug and logging
            logger.log_metrics(batch=batch, model=correspondences)
            if (iter_step % self.save_interval) == 0:
                logger.log_3d_points(batch=batch, model=correspondences)
                logger.log_render(batch=batch, model=correspondences)
                logger.log_input_batch(batch=batch, model=correspondences)
                logger.log_loss(batch=batch, model=correspondences)
                if model.init_mode == "flame":
                    model.debug_params(batch=batch)

    def optimize(self):
        raise NotImplementedError


class JointTrainer(BaseTrainer):
    def __init__(self, init_idxs: list[int] = [], **kwargs):
        super().__init__(**kwargs)
        assert len(init_idxs) > 0
        self.init_idxs = init_idxs
        self.frames = init_idxs

    def optimize(self):
        self.logger.prefix = "joint"
        self.optimizer.reset()
        self.coarse2fine.reset()
        self.datamodule.update_idxs(self.init_idxs)
        self.optimize_loop()


class SequentialTrainer(BaseTrainer):
    def __init__(self, start_frame: int = 0, end_frame: int = 126, **kwargs):
        super().__init__(**kwargs)
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.frames = list(range(self.start_frame, self.end_frame))

    def optimize(self):
        for frame_idx in tqdm(range(self.start_frame, self.end_frame)):
            self.logger.prefix = "sequential"
            self.optimizer.reset()
            self.coarse2fine.reset()
            self.datamodule.update_idxs([frame_idx])

            if frame_idx != 0:
                self.model.init_frame(frame_idx)
            self.optimize_loop()
