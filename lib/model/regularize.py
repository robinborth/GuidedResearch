import torch
import torch.nn as nn

from lib.model.common import MLP


class MLPRegularizeModule(nn.Module):
    def __init__(
        self,
        features: int = 100,
        depth: int = 3,
        unet_features: int = 64,
        unet_depth: int = 4,
        unet_size: int = 256,
        flame_expression_params: int = 50,
    ):
        super().__init__()

        factor = 2**unet_depth
        unet_size = int(unet_size // factor)
        unet_features = int(unet_features * factor)

        self.conv = nn.Conv2d(
            in_channels=unet_features,
            out_channels=1,
            kernel_size=1,
        )
        self.mlp = MLP(
            in_dim=(unet_size * unet_size),
            out_dim=flame_expression_params,
            hidden_dim=features,
            num_layers=depth,
        )

    def forward(self, params: dict[str, torch.Tensor], latent: torch.Tensor):
        # flat latent
        x = self.conv(latent)  # (B, 1, H, W)
        x = x.flatten(start_dim=1)
        # add params information
        x = self.mlp(x)
        x = x + params["expression_params"]
        return dict(expression_params=x)


class DummyRegularizeModule(nn.Module):
    def forward(self, params: dict[str, torch.Tensor], **kwargs):
        out = {}
        for key, value in params.items():
            out[key] = torch.zeros_like(value)
        return out
