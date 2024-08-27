import logging

import torch
from tqdm import tqdm

from lib.data.datamodule import DPHMDataModule
from lib.data.synthesis import generate_params
from lib.optimizer.framework import OptimizerFramework
from lib.tracker.scheduler import CoarseToFineScheduler, OptimizerScheduler
from lib.utils.progress import close_progress, reset_progress

log = logging.getLogger()


class JointTracker:
    def __init__(
        self,
        datamodule: DPHMDataModule,
        optimizer: OptimizerFramework,
        scheduler: OptimizerScheduler,
        coarse2fine: CoarseToFineScheduler,
        max_iters: int = 1,
        max_optims: int = 1,
        init_idxs: list[int] = [],
        default_params: dict = {},
    ):
        self.mode = "joint"
        self.datamodule = datamodule
        self.coarse2fine = coarse2fine
        self.scheduler = scheduler
        self.max_iters = max_iters
        self.max_optims = max_optims
        self.optimizer = optimizer
        self.init_idxs = init_idxs
        self.default_params = default_params
        assert len(init_idxs) > 0

    def outer_progress(self):
        return tqdm(total=self.max_iters, desc="Outer Loop", position=1)

    def inner_progress(self):
        return tqdm(total=self.max_optims, desc="Inner Loop", leave=True, position=2)

    def optimize(self):
        # build the batch
        batch = {}
        batch["params"] = generate_params(
            self.optimizer.flame,
            window_size=len(self.init_idxs),
            default=self.default_params,
        )
        batch["outer_progress"] = self.outer_progress()
        batch["inner_progress"] = self.inner_progress()
        batch["max_iters"] = self.max_iters
        batch["max_optims"] = self.max_optims
        batch["mode"] = self.mode
        batch["coarse2fine"] = self.coarse2fine
        batch["scheduler"] = self.scheduler
        batch["datamodule"] = self.datamodule

        self.datamodule.update_idxs(self.init_idxs)
        with torch.no_grad():
            out = self.optimizer(batch)
        close_progress([batch["outer_progress"], batch["inner_progress"]])
        return out["params"]


class SequentialTracker:
    def __init__(
        self,
        datamodule: DPHMDataModule,
        optimizer: OptimizerFramework,
        scheduler: OptimizerScheduler,
        coarse2fine: CoarseToFineScheduler,
        max_iters: int = 1,
        max_optims: int = 1,
        kernel_size: int = 1,
        stride: int = 1,
        dilation: int = 1,
        start_frame: int = 0,
        end_frame: int = 126,
        default_params: dict = {},
    ):
        self.mode = "sequential"
        self.final_video = True
        self.datamodule = datamodule
        self.coarse2fine = coarse2fine
        self.scheduler = scheduler
        self.max_iters = max_iters
        self.max_optims = max_optims
        self.optimizer = optimizer
        self.default_params = {
            k: v[0].detach().cpu().tolist() for k, v in default_params.items()
        }
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.kernal_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        full_frames = [i for k in self.frame_idxs_iter() for i in k]
        self.frames = sorted(set(full_frames))

    def frame_progress(self):
        total = len(list(self.frame_idxs_iter()))
        return tqdm(total=total, desc="Frame Loop", position=0)

    def outer_progress(self):
        return tqdm(total=self.max_iters, desc="Outer Loop", position=1)

    def inner_progress(self):
        return tqdm(total=self.max_optims, desc="Inner Loop", leave=True, position=2)

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

    def init_frames(self, init_params: dict):
        return init_params

    def optimize(self):
        store = []

        frame_progress = self.frame_progress()
        outer_progress = self.outer_progress()
        inner_progress = self.inner_progress()

        # build the batch
        batch = {}
        batch["params"] = generate_params(
            self.optimizer.flame,
            window_size=self.kernal_size,
            default=self.default_params,
        )
        batch["outer_progress"] = outer_progress
        batch["inner_progress"] = inner_progress
        batch["max_iters"] = self.max_iters
        batch["max_optims"] = self.max_optims
        batch["mode"] = self.mode
        batch["datamodule"] = self.datamodule
        batch["coarse2fine"] = self.coarse2fine
        batch["scheduler"] = self.scheduler

        for frame_idxs in self.frame_idxs_iter():
            reset_progress(outer_progress, self.max_iters)
            self.datamodule.update_idxs(frame_idxs)
            with torch.no_grad():
                out = self.optimizer(batch)
            # updatae the params
            batch["params"] = out["params"]
            store.append(dict(params=out["params"], frame_idx=frame_idxs))
            frame_progress.update(1)

        # close the progresses
        close_progress([frame_progress, outer_progress, inner_progress])

        return store


# class PCGSamplingTrainer(BaseTrainer):
#     def __init__(self, init_idxs: list[int] = [], max_samplings: int = 1000, **kwargs):
#         super().__init__(**kwargs)
#         self.mode = "pcg"
#         assert len(init_idxs) > 0
#         self.init_idxs = init_idxs
#         self.frames = init_idxs
#         self.max_samplings = max_samplings

#     def sampling_progress(self):
#         return tqdm(total=self.max_samplings, desc="Sampling Loop", position=0)

#     def optimize(self):
#         sampling_progress = self.sampling_progress()
#         outer_progress = self.outer_progress()
#         inner_progress = self.inner_progress()

#         for _ in range(self.max_samplings):
#             self.model.init_params_with_config(self.model.init_config)

#             self.scheduler.freeze(self.model)
#             self.logger.mode = self.mode
#             self.scheduler.reset()
#             self.coarse2fine.reset()
#             self.datamodule.update_idxs(self.init_idxs)

#             self.optimize_loop(outer_progress, inner_progress)

#             sampling_progress.update(1)

#         self.close_progress([sampling_progress, outer_progress, inner_progress])
