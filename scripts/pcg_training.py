import logging
from collections import defaultdict
from pathlib import Path
from typing import List

import hydra
import lightning as L
import matplotlib.pyplot as plt
import torch
import wandb
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from lib.optimizer.solver import (
    ConditionNet,
    IdentityConditionNet,
    JaccobiConditionNet,
    PCGSolver,
)
from lib.utils.config import instantiate_callbacks, log_hyperparameters, set_configs

log = logging.getLogger()


def gather_convergence_stats(output, keys=None):
    stats = defaultdict(list)
    for batch in output:
        for key, value in batch.items():
            if keys is None or key in keys:
                stats[key].append(value)
    for key, value in stats.items():
        stats[key] = torch.cat(value, dim=-1).mean(dim=-1)
    return stats


def visuailize_convergence(outputs: list, labels: list, markers: list, path: Path):
    plt.figure(figsize=(10, 6))
    for idx, stats in enumerate(outputs):
        relres = stats["relres_norms"]
        iterations = range(len(relres))
        plt.plot(iterations, relres, label=labels[idx], marker=markers[idx])

    # Adding titles and labels
    plt.title("Relative Residual Norms Comparison")
    plt.xlabel("Iterations")
    plt.ylabel("Relative Residual Norms")

    # Set y-axis to log scale
    plt.yscale("log")

    # Set y-axis limit to start from 10^-8
    plt.ylim(bottom=1e-7)

    # Add a horizontal red line at 10^-6
    plt.axhline(y=1e-6, color="red", linestyle="--", label="Convergence at $10^{-6}$")

    # Show legend
    plt.legend()

    # Show grid
    plt.grid(True)

    # save the figure
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)


@hydra.main(version_base=None, config_path="../conf", config_name="pcg_training")
def train(cfg: DictConfig) -> None:
    log.info("==> loading config ...")
    cfg = set_configs(cfg)

    log.info(f"==> initializing datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"==> initializing model <{cfg.model._target_}>")
    model: PCGSolver = hydra.utils.instantiate(cfg.model)

    log.info("==> initializing logger ...")
    logger = hydra.utils.instantiate(cfg.logger)
    logger.watch(model, log="all")

    log.info("==> initializing callbacks ...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info(f"==> initializing trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }
    if logger:
        log.info("==> logging hyperparameters ...")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("==> start training ...")
        trainer.fit(model=model, datamodule=datamodule)

    if cfg.get("eval"):
        if cfg.get("ckpt_path"):
            log.warning("==> using checkpoint for testing...")
            model = PCGSolver.load_from_checkpoint(cfg.ckpt_path)
        else:
            log.warning("==> using current weights for testing...")
        log.info("==> loading condition networks ...")
        condition_nets: list[ConditionNet] = [
            IdentityConditionNet(),
            JaccobiConditionNet(),
            model.condition_net,
        ]

        log.info("==> start evaluating convergence ...")
        model.max_iter = cfg.eval_max_iter
        datamodule.hparams["batch_size"] = cfg.eval_batch_size
        datamodule.setup("all")
        for split in cfg.eval_splits:
            outputs = []
            for condition_net in condition_nets:
                log.info(f"==> evaluate {condition_net.name} ({split})...")
                model.condition_net = condition_net  # type: ignore
                loader = getattr(datamodule, f"{split}_dataloader")
                output = trainer.predict(model, dataloaders=[loader()])
                stats = gather_convergence_stats(output, ["relres_norms"])
                outputs.append(stats)

            path = Path(cfg.paths.output_dir) / f"pcg_convergence_{split}.png"
            visuailize_convergence(
                outputs=outputs,
                labels=["identity", "jaccobi", "pcg"],
                markers=["o", "o", "o"],
                path=path,
            )
            wandb.log({f"pcg_convergence_{split}": wandb.Image(str(path))})

        log.info("==> start evaluating efficiency ...")
        model.max_iter = 50  # pick large enough so we don't converge
        model.check_convergence = True
        datamodule.hparams["batch_size"] = 1  # realistic time measurement

        stats = []
        for condition_net in condition_nets:
            log.info(f"==> evaluate {condition_net.name} ...")
            model.condition_net = condition_net  # type: ignore
            output = trainer.predict(model, datamodule=datamodule)
            stat = gather_convergence_stats(output, ["cond", "iters", "measure_time"])
            stat["name"] = condition_net.name
            stats.append(stat)

        columns = ["name", "cond", "iters", "measure_time"]
        data = []
        for stat in stats:
            data.append([stat[column] for column in columns])
        table = wandb.Table(columns=columns, data=data)
        wandb.log({"pcg_efficiency": table})

        columns = ["cond", "iters", "measure_time"]
        for stat in stats:
            if model.condition_net.name == stat["name"]:
                metrics = {f"pcg_{k}": v for k, v in stat.items()}
                wandb.log(metrics)

    # closing wandb
    wandb.finish()


if __name__ == "__main__":
    train()
