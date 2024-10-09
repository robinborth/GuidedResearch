import logging

import hydra
import torch
from omegaconf import DictConfig
from tqdm.notebook import tqdm

import wandb
from lib.data.loader import load_intrinsics
from lib.optimizer.framework import NeuralOptimizer
from lib.rasterizer import Rasterizer
from lib.renderer.camera import Camera
from lib.renderer.renderer import Renderer
from lib.tracker.timer import TimeTracker
from lib.utils.config import load_config
from lib.utils.progress import close_progress, reset_progress


def path_to_abblation(path):
    return "_".join(path.split("/")[-3].split("_")[1:])


def eval_iterations(optimizer, datamodule, N: int = 1, mode="iters"):
    optimizer.max_iters = 1
    optimizer.max_optims = 1

    outer_progress = tqdm(total=N + 1, desc="Iter Loop", position=0)
    total_evals = len(datamodule.val_dataset)
    inner_progress = tqdm(total=total_evals, desc="Eval Loop", leave=True, position=1)

    iters_p_loss = {}
    iters_v_loss = {}
    iters_time = {}

    # initial evaluation no optimization
    reset_progress(inner_progress, total_evals)
    p_loss = []
    v_loss = []
    for batch in datamodule.val_dataloader():
        with torch.no_grad():
            batch = optimizer.transfer_batch_to_device(batch, "cuda", 0)
            out = optimizer(batch)
            out["params"] = batch["init_params"]
            loss_info = optimizer.compute_loss(batch=batch, out=out)
            p_loss.append(loss_info["loss_param"])
            v_loss.append(loss_info["loss_vertices"])
        inner_progress.update(1)
    iters_p_loss[0] = torch.stack(p_loss)
    iters_v_loss[0] = torch.stack(v_loss)
    iters_time[0] = torch.zeros_like(iters_p_loss[0])
    outer_progress.update(1)

    # evaluation after some optimization
    for iters in range(1, N + 1):
        reset_progress(inner_progress, total_evals)
        if mode == "iters":
            optimizer.max_iters = iters
        else:
            optimizer.max_optims = iters
        time_tracker = TimeTracker()
        p_loss = []
        v_loss = []
        for batch in datamodule.val_dataloader():
            with torch.no_grad():
                batch = optimizer.transfer_batch_to_device(batch, "cuda", 0)
                time_tracker.start("optimize")
                out = optimizer(batch)
                time_tracker.stop("optimize")
                loss_info = optimizer.compute_loss(batch=batch, out=out)
                p_loss.append(loss_info["loss_param"])
                v_loss.append(loss_info["loss_vertices"])
            inner_progress.update(1)
        iters_p_loss[iters] = torch.stack(p_loss)
        iters_v_loss[iters] = torch.stack(v_loss)
        iters_time[iters] = torch.stack(
            [torch.tensor(t.time_ms) for t in list(time_tracker.tracks.values())[0]]
        )
        outer_progress.update(1)
    close_progress([outer_progress, inner_progress])
    return iters_p_loss, iters_v_loss, iters_time


def load_flame_renderer():
    # instanciate similar to training
    cfg = load_config("train", ["data=synthetic"])
    K = load_intrinsics(data_dir=cfg.data.intrinsics_dir, return_tensor="pt")
    camera = Camera(
        K=K,
        width=cfg.data.width,
        height=cfg.data.height,
        near=cfg.data.near,
        far=cfg.data.far,
        scale=cfg.data.scale,
    )
    rasterizer = Rasterizer(width=camera.width, height=camera.height)
    renderer = Renderer(rasterizer=rasterizer, camera=camera)
    flame = hydra.utils.instantiate(cfg.model)
    return flame, renderer


def load_neural_optimizer(flame, renderer, path, override=[]):
    o = ["data=synthetic"] + override
    cfg = load_config("train", o)
    correspondence = hydra.utils.instantiate(cfg.correspondence)
    weighting = hydra.utils.instantiate(cfg.weighting)
    residuals = hydra.utils.instantiate(cfg.residuals)
    regularize = hydra.utils.instantiate(cfg.regularize)
    neural_optimizer = NeuralOptimizer.load_from_checkpoint(
        path,
        renderer=renderer,
        flame=flame,
        correspondence=correspondence,
        regularize=regularize,
        residuals=residuals,
        weighting=weighting,
    )
    return neural_optimizer


def load_icp_optimizer(flame, renderer, overrides):
    o = ["data=synthetic", "optimizer.output_dir=none"] + overrides
    cfg = load_config("train", o)
    correspondence = hydra.utils.instantiate(cfg.correspondence)
    weighting = hydra.utils.instantiate(cfg.weighting)
    residuals = hydra.utils.instantiate(cfg.residuals)
    optimizer = hydra.utils.instantiate(cfg.optimizer)
    regularize = hydra.utils.instantiate(cfg.regularize)
    icp_optimizer = hydra.utils.instantiate(
        cfg.framework,
        flame=flame,
        logger=None,
        renderer=renderer,
        correspondence=correspondence,
        regularize=regularize,
        residuals=residuals,
        optimizer=optimizer,
        weighting=weighting,
    )
    return icp_optimizer.to("cuda")


# setup the datamodule
def load_datamodule(renderer, start_frame, end_frame):
    cfg = load_config("train", ["data=synthetic"])
    datamodule = hydra.utils.instantiate(
        cfg.data,
        renderer=renderer,
        val_dataset=dict(
            start_frame=start_frame,
            end_frame=end_frame,
        ),
    )
    datamodule.setup("fit")
    return datamodule


def main():

    # settings
    N = 3
    start_frame = None 
    end_frame = None

    # checkpoints
    ours = "/home/borth/GuidedResearch/logs/2024-10-06/06-55-55_abblation_ours_ckpt/checkpoints/last.ckpt"
    wo_neural_prior = "/home/borth/GuidedResearch/logs/2024-10-04/22-44-20_abblation_wo_neural_prior/checkpoints/last.ckpt"
    w_single_corresp = "/home/borth/GuidedResearch/logs/2024-10-03/09-54-41_abblation_w_single_corresp/checkpoints/last.ckpt"
    w_single_optim = "/home/borth/GuidedResearch/logs/2024-10-06/12-55-40_abblation_w_single_optim/checkpoints/last.ckpt"
    wo_neural_weights = "/home/borth/GuidedResearch/logs/2024-10-03/09-54-41_abblation_wo_neural_weights/checkpoints/last.ckpt"

    # loadings
    times = {}
    p_losses = {}
    v_losses = {}
    flame, renderer = load_flame_renderer()
    datamodule = load_datamodule(renderer, start_frame, end_frame)

    path = ours
    optimizer = load_neural_optimizer(flame, renderer, path)
    p_loss, v_loss, time = eval_iterations(optimizer, datamodule, N=N, mode="iters")
    key = path_to_abblation(path)
    times[key] = time[N].median().item()
    p_losses[key] = p_loss[N].mean().item()
    v_losses[key] = v_loss[N].mean().item()
    print(
        f"{key}: p_loss={p_losses[key]:.03f} v_loss={v_losses[key]:.03f} time={times[key]:.03f}ms"
    )

    path = wo_neural_weights
    optimizer = load_neural_optimizer(
        flame, renderer, path, ["weighting.dummy_weight=True"]
    )
    p_loss, v_loss, time = eval_iterations(optimizer, datamodule, N=N, mode="iters")
    key = path_to_abblation(path)
    times[key] = time[N].median().item()
    p_losses[key] = p_loss[N].mean().item()
    v_losses[key] = v_loss[N].mean().item()
    print(
        f"{key}: p_loss={p_losses[key]:.03f} v_loss={v_losses[key]:.03f} time={times[key]:.03f}ms"
    )

    path = wo_neural_prior
    optimizer = load_neural_optimizer(flame, renderer, path, ["regularize=dummy"])
    p_loss, v_loss, time = eval_iterations(optimizer, datamodule, N=N, mode="iters")
    key = path_to_abblation(path)
    times[key] = time[N].median().item()
    p_losses[key] = p_loss[N].mean().item()
    v_losses[key] = v_loss[N].mean().item()
    print(
        f"{key}: p_loss={p_losses[key]:.03f} v_loss={v_losses[key]:.03f} time={times[key]:.03f}ms"
    )

    path = w_single_corresp
    optimizer = load_neural_optimizer(flame, renderer, path)
    p_loss, v_loss, time = eval_iterations(optimizer, datamodule, N=N, mode="optims")
    key = path_to_abblation(path)
    times[key] = time[N].median().item()
    p_losses[key] = p_loss[N].mean().item()
    v_losses[key] = v_loss[N].mean().item()
    print(
        f"{key}: p_loss={p_losses[key]:.03f} v_loss={v_losses[key]:.03f} time={times[key]:.03f}ms"
    )

    path = w_single_optim
    optimizer = load_neural_optimizer(flame, renderer, path)
    p_loss, v_loss, time = eval_iterations(optimizer, datamodule, N=N, mode="iters")
    key = "abblation_wo_end_to_end"
    times[key] = time[N].median().item()
    p_losses[key] = p_loss[N].mean().item()
    v_losses[key] = v_loss[N].mean().item()
    print(
        f"{key}: p_loss={p_losses[key]:.03f} v_loss={v_losses[key]:.03f} time={times[key]:.03f}ms"
    )
    key = path_to_abblation(path)
    times[key] = time[1].median().item()
    p_losses[key] = p_loss[1].mean().item()
    v_losses[key] = v_loss[1].mean().item()
    print(
        f"{key}: p_loss={p_losses[key]:.03f} v_loss={v_losses[key]:.03f} time={times[key]:.03f}ms"
    )

    wandb.init(project="guided-research", entity="robinborth")
    wandb.log({f"p/{k}": v for k, v in p_losses.items()})
    wandb.log({f"v/{k}": v for k, v in v_losses.items()})
    wandb.log({f"t/{k}": v for k, v in times.items()})


if __name__ == "__main__":
    main()
