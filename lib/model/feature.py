import torch
import torch.nn as nn

from lib.model.common import CNNMLP


class DeepFeatureModule(nn.Module):
    def __init__(
        self,
        out_channels: int = 128,
        hidden_channels: int = 100,
        kernal_size: int = 3,
        num_layers: int = 1,
        device: str = "cuda",
    ):
        super().__init__()
        self.cnn = CNNMLP(
            in_channels=3,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            kernal_size=kernal_size,
            num_layers=num_layers,
        )
        self.to(device)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 3, 1, 2)  # (B, 6, H, W)
        return self.cnn(x).permute(0, 2, 3, 1)  # (B, W, H, 1)
