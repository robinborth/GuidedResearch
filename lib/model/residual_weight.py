from typing import Any

import torch
import torch.nn as nn


class ResidualWeightModule(nn.Module):
    def __init__(
        self,
        in_channels: int = 5,  # source_depth, target_depth, target_normal
        out_channels: int = 1,  # weight
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

    def forward(self, x):  # (B, C, W, H)
        return torch.exp(self.cnn(x)).squeeze(1)  # (B, W, H)
