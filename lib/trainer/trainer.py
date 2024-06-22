import logging

from lib.model.flame import FLAME
from lib.trainer.logger import FlameLogger

log = logging.getLogger()


class Trainer:
    def __init__(
        self,
        model: FLAME,
        logger: FlameLogger,
        # loop settings
        max_iters: int = 25,
        max_optims: int = 100,
        # optimizer settings
        optimizer: str = "levenberg_marquardt",
        copy_optimizer_state: bool = False,
        # logging settings
        save_interval: int = 1,
        # convergence tests
        check_convergence: bool = False,
        convergence_threshold: float = 1e-10,
        min_tracker_steps: int = 2,
        max_tracker_steps: int = 5,
    ):
        self.model = model
        self.logger = logger
        self.max_iters = max_iters
        self.max_optims = max_optims
        self.optimizer = optimizer
        self.copy_optimizer_state = copy_optimizer_state
        self.save_interval = save_interval
        self.check_convergence = check_convergence
        self.convergence_threshold = convergence_threshold
        self.min_tracker_steps = min_tracker_steps
        self.max_tracker_steps = max_tracker_steps

    def optimize(self):
        pass

    def inner_loop(self, iter_step: int):
        # inner optimization loop
        loss_tracker: list[float] = []
        for optim_step in range(self.max_optims):
            # state
            log.info(f"optim_step: {optim_step}")
            self.logger.optim_step = optim_step
            self.logger.global_step = iter_step * self.max_optims + optim_step + 1

            # optimize step
            if os.requires_jacobian(self.optimizer):
                loss = optimizer.newton_step(
                    model.loss_closure(batch, correspondences),
                    model.jacobian_closure(batch, correspondences, optimizer),
                )
            elif os.requires_loss(optimizer):
                loss = optimizer.step(model.loss_closure(batch, correspondences))
            else:
                optimizer.zero_grad()
                loss = model.loss_step(batch, correspondences)
                loss.backward()
                optimizer.step()

            # logging
            logger.log("loss/point2plane", loss)
            logger.log(f"loss/{iter_step:03}/point2plane", loss)
            logger.log_gradients(optimizer)

            # check for convergence
            if cfg.trainer.check_convergence:
                loss_tracker.append(loss)
                if len(loss_tracker) >= cfg.trainer.min_tracker_steps:
                    tracks = loss_tracker[-cfg.trainer.max_tracker_steps :]
                    criterion = torch.tensor(tracks).std()
                    logger.log("convergence/std", criterion)
                    logger.log(f"convergence/{iter_step:03}/std", criterion)
                    if criterion < cfg.trainer.convergence_threshold:
                        break
