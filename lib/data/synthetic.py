from collections import defaultdict

import torch

from lib.model import Flame


def generate_params(
    flame: Flame,
    window_size: int = 1,
    default: dict = {},
    sigmas: dict = {},
    select: list[str] | None = None,
):
    # default params
    base_params = flame.generate_default_params()
    for p_name, param in default.items():
        base_params[p_name] = torch.tensor([param], device=flame.device)

    params = defaultdict(list)
    for p_name in flame.global_params:
        eps = torch.randn_like(base_params[p_name]) * sigmas.get(p_name, 0.0)
        params[p_name] = base_params[p_name] + eps
    for p_name in flame.local_params:
        for _ in range(window_size):
            eps = torch.randn_like(base_params[p_name]) * sigmas.get(p_name, 0.0)
            param = base_params[p_name] + eps
            params[p_name].append(param)
    for p_name in flame.local_params:
        params[p_name] = torch.cat(params[p_name], dim=0)

    if select is None:
        return params

    return {k: v for k, v in params.items() if k in select}


def generate_synthetic_params(
    flame: Flame,
    window_size: int = 1,
    default: dict = {},
    sigmas: dict = {},
    select: list[str] | None = None,
    sparsity: dict = {},
):
    full_params = generate_params(
        flame=flame,
        window_size=window_size,
        default=default,
        sigmas=sigmas,
        select=select,
    )
    params = {}
    for key, value in full_params.items():
        if key == "expression_params":
            value += torch.abs(value) * 0.5
        spars = sparsity.get(key, 0.0)
        zero_mask = torch.rand_like(value) < spars
        value[zero_mask] = 0.0
        params[key] = value
    return params
