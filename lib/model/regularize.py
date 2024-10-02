import torch
import torch.nn as nn

from lib.model.common import MLP


class MLPRegularizeModule(nn.Module):
    def __init__(
        self,
        features: int = 100,
        depth: int = 3,
        latent_scale: int = 4,
        unet_features: int = 64,
        unet_depth: int = 4,
        unet_size: int = 256,
        expression_params: int = 50,
        shape_params: int = 100,
        dummy_weight: bool = False,
        dummy_delta: bool = False,
    ):
        super().__init__()

        self.dummy_weight = dummy_weight
        self.dummy_delta = dummy_delta

        factor = 2**unet_depth
        unet_size = int(unet_size // factor)
        unet_features = int(unet_features * factor)

        # merge to one shared latent
        self.conv = nn.Conv2d(
            in_channels=unet_features,
            out_channels=latent_scale,
            kernel_size=1,
        )
        self.mlp = self._mlp_body(
            in_dim=(unet_size * unet_size * latent_scale) + 59,
            features=features,
            depth=depth,
        )

        # each param gets own projection layer for the delta params
        mlp_delta_heads = dict(
            shape_params=self._mlp_head(features, shape_params),
            expression_params=self._mlp_head(features, expression_params),
            transl=self._mlp_head(features, 3),
            global_pose=self._mlp_head(features, 3),
            neck_pose=self._mlp_head(features, 3),
        )
        self.mlp_delta_heads = nn.ModuleDict(mlp_delta_heads)

        # each param gets own projection layer for the weights
        mlp_weight_heads = dict(
            shape_params=self._mlp_head(features, shape_params),
            expression_params=self._mlp_head(features, expression_params),
            transl=self._mlp_head(features, 3),
            global_pose=self._mlp_head(features, 3),
            neck_pose=self._mlp_head(features, 3),
        )
        self.mlp_weight_heads = nn.ModuleDict(mlp_weight_heads)

    def _mlp_body(self, in_dim, features, depth):
        return MLP(
            in_dim=in_dim,
            out_dim=features,
            hidden_dim=features,
            num_layers=depth,
        )

    def _mlp_head(self, features, out_dim):
        return MLP(
            in_dim=features,
            out_dim=out_dim,
            hidden_dim=features,
            num_layers=0,
        )

    def forward(self, params: dict[str, torch.Tensor], latent: torch.Tensor):
        # global shared feature
        x = self.conv(latent)  # (B, 1, H, W)
        x = x.flatten(start_dim=1)

        x_params = torch.cat(list(params.values()), dim=-1)
        x = torch.cat([x_params, x], dim=-1)
        x = self.mlp(x)

        # compute the per parameter weights and deltas
        deltas = {}  # type: ignore
        priors = {}
        weights = {}
        for key, value in params.items():
            if self.dummy_weight:
                weights[key] = torch.ones_like(value)
            else:
                weight = self.mlp_weight_heads[key](x)
                weights[key] = torch.nn.functional.elu(weight) + 1.0
            if self.dummy_delta:
                priors[key] = torch.zeros_like(value)
                deltas[key] = None  # type: ignore
            else:
                deltas[key] = self.mlp_delta_heads[key](x)
                priors[key] = deltas[key] + value
        return dict(priors=priors, deltas=deltas, weights=weights)


class DummyRegularizeModule(nn.Module):
    def forward(self, params: dict[str, torch.Tensor], **kwargs):
        deltas = {}  # type: ignore
        priors = {}
        weights = {}
        for key, value in params.items():
            weights[key] = torch.ones_like(value)
            priors[key] = torch.zeros_like(value)
            deltas[key] = None  # type: ignore
        return dict(priors=priors, deltas=deltas, weights=weights)
