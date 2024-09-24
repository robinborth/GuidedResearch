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
        expression_params: int = 50,
        shape_params: int = 100,
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
        self.mlp_heads = nn.ModuleDict(
            {
                "shape_params": MLP(
                    in_dim=(unet_size * unet_size),
                    out_dim=shape_params,
                    hidden_dim=features,
                    num_layers=depth,
                ),
                "expression_params": MLP(
                    in_dim=(unet_size * unet_size),
                    out_dim=expression_params,
                    hidden_dim=features,
                    num_layers=depth,
                ),
                "transl": MLP(
                    in_dim=(unet_size * unet_size),
                    out_dim=3,
                    hidden_dim=features,
                    num_layers=depth,
                ),
                "global_pose": MLP(
                    in_dim=(unet_size * unet_size),
                    out_dim=3,
                    hidden_dim=features,
                    num_layers=depth,
                ),
                "neck_pose": MLP(
                    in_dim=(unet_size * unet_size),
                    out_dim=3,
                    hidden_dim=features,
                    num_layers=depth,
                ),
            }
        )

    def forward(self, params: dict[str, torch.Tensor], latent: torch.Tensor):
        # flat latent
        x = self.conv(latent)  # (B, 1, H, W)
        x = x.flatten(start_dim=1)
        deltas = {}
        regularize = {}
        for key, value in params.items():
            delta = self.mlp_heads[key](x)
            regularize[key] = delta + value
            deltas[key] = delta
        return dict(regularize=regularize, deltas=deltas)


class DummyRegularizeModule(nn.Module):
    def forward(self, params: dict[str, torch.Tensor], **kwargs):
        deltas = {}  # type: ignore
        regularize = {}
        for key, value in params.items():
            regularize[key] = torch.zeros_like(value)
            deltas[key] = None  # type: ignore
        return dict(regularize=regularize, deltas=deltas)
