import logging

import torch
from tqdm import tqdm

from lib.data.datamodule import DPHMDataModule
from lib.model.flame import FLAME
from lib.trainer.logger import FlameLogger
from lib.trainer.scheduler import CoarseToFineScheduler, OptimizerScheduler

log = logging.getLogger()


class Trainer:
    def __init__(
        self,
        model: FLAME,
        logger: FlameLogger,
        # loop settings
        max_iters: int = 25,
        max_optims: int = 100,
        # logging settings
        save_interval: int = 1,
        prefix: str = "",
        # convergence tests
        check_convergence: bool = False,
        convergence_threshold: float = 1e-10,
        min_tracker_steps: int = 2,
        max_tracker_steps: int = 5,
    ):
        # set the state
        self.model = model
        self.logger = logger

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
        self.prefix = prefix

    def optimize(
        self,
        datamodule: DPHMDataModule,
        optimizer_scheduler: OptimizerScheduler,
        coarse_to_fine_scheduler: CoarseToFineScheduler,
    ):
        logger = self.logger
        model = self.model

        # outer optimization loop
        optimizer_scheduler.freeze(model)
        for iter_step in tqdm(range(self.max_iters)):
            logger.iter_step = iter_step

            # fetch single batch
            coarse_to_fine_scheduler.schedule(
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
                logger.log("loss/point2plane", loss)
                logger.log(f"loss/{iter_step:03}/point2plane", loss)
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
