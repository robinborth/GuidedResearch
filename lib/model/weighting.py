import torch
import torch.nn as nn

from lib.model.common import CNNMLP


class ResidualWeightModule(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 100,
        kernal_size: int = 3,
        num_layers: int = 1,
        mode: str = "point",  # "point", "normal", "point_normal"
        max_weight: float = 100.0,
        device: str = "cuda",
    ):
        super().__init__()

        self.max_weight = max_weight

        self.mode = mode
        assert mode in ["point", "normal", "point_normal"]
        in_channels = 6
        if mode == "point_normal":
            in_channels = 12

        self.cnn = CNNMLP(
            in_channels=in_channels,
            out_channels=1,
            hidden_channels=hidden_channels,
            kernal_size=kernal_size,
            num_layers=num_layers,
        )
        self.to(device)

    def forward(
        self,
        s_point: torch.Tensor,
        s_normal: torch.Tensor,
        t_point: torch.Tensor,
        t_normal: torch.Tensor,
    ):
        if self.mode == "point_normal":  # (B, H, W, 12)
            x = torch.cat([s_point, s_normal, t_point, t_normal], dim=-1)
        elif self.mode == "point":
            x = torch.cat([s_point, t_point], dim=-1)  # (B, H, W, 6)
        elif self.mode == "normal":
            x = torch.cat([s_normal, t_normal], dim=-1)  # (B, H, W, 6)
        else:
            raise AttributeError(f"No {self.mode} that works.")

        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        x = torch.exp(self.cnn(x))  # (B, W, H, 1)
        x = torch.min(x, torch.tensor(self.max_weight))
        return x


class DummyWeightModule(nn.Module):
    def forward(self, s_point: torch.Tensor, **kwargs):
        B, H, W, _C = s_point.shape
        return torch.ones((B, H, W), device=s_point.device)
