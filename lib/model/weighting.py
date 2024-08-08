import torch
import torch.nn as nn

from lib.model.common import CNNMLP


class ResidualWeightModule(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 100,
        kernal_size: int = 3,
        num_layers: int = 1,
        device: str = "cuda",
    ):
        super().__init__()
        self.cnn = CNNMLP(
            in_channels=6,
            out_channels=1,
            hidden_channels=hidden_channels,
            kernal_size=kernal_size,
            num_layers=num_layers,
        )
        self.to(device)

    def forward(self, s_point: torch.Tensor, t_point: torch.Tensor):
        x = torch.cat([s_point, t_point], dim=-1)  # (B, H, W, 6)
        x = x.permute(0, 3, 1, 2)  # (B, 6, H, W)
        return torch.exp(self.cnn(x))  # (B, W, H, 1)


class DummyWeightModule(nn.Module):
    def forward(self, s_point: torch.Tensor, t_point: torch.Tensor):
        B, H, W, _C = s_point.shape
        return torch.ones((B, H, W), device=s_point.device)
