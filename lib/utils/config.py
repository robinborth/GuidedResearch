from typing import List

from hydra import compose, initialize
from hydra.utils import instantiate
from lightning import Callback
from omegaconf import DictConfig, OmegaConf

from lib.utils.logger import create_logger

log = create_logger("config")


def instantiate_callbacks(callbacks_cfg: DictConfig | dict) -> List[Callback]:
    callbacks: List[Callback] = []

    if callbacks_cfg is None:
        log.info("No callbacks specified.")
        return callbacks

    for callback in callbacks_cfg.values():
        if "_target_" in callback.keys():
            callbacks.append(instantiate(callback))

    return callbacks


def load_config(config_name: str, overrides: list = []) -> DictConfig:
    """Loads the hydra config via code.

    Args:
        config_name (str): The name of the configuration.
        overrides (list): List of overrides to apply.

    Returns:
        DictConfig: The initialized config.
    """
    with initialize(config_path="../../conf", version_base=None):
        return compose(config_name=config_name, overrides=overrides)
