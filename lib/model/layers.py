from typing import Any

import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int = 1,
        out_dim: int = 1,
        hidden_dim: int = 1,
        num_layers: int = 1,
    ):
        super().__init__()
        layers: list[Any] = []
        # input layers
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        # hidden layers
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        # output layer
        layers.append(nn.Linear(hidden_dim, out_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
