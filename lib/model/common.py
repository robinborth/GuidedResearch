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


class CNNMLP(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,  # s_point, t_point
        out_channels: int = 1,  # weights
        hidden_channels: int = 1,
        kernal_size: int = 3,
        num_layers: int = 1,
    ):
        super().__init__()

        layers: list[Any] = []
        # input layers
        layers.append(
            nn.Conv2d(in_channels, hidden_channels, kernal_size, padding="same")
        )
        layers.append(nn.ReLU())
        # hidden layers
        for _ in range(num_layers):
            layers.append(
                nn.Conv2d(hidden_channels, hidden_channels, kernal_size, padding="same")
            )
            layers.append(nn.ReLU())
        # output layer
        layers.append(
            nn.Conv2d(hidden_channels, out_channels, kernal_size, padding="same")
        )

        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.cnn(x).squeeze(1)  # (B, W, H)
