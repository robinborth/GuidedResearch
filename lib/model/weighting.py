import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.model.common import CNNMLP


class WeightModule(nn.Module):
    def compute_in_channels(self, mode: str = "point_normal"):
        assert mode in ["point", "normal", "point_normal"]
        in_channels = 6
        if mode == "point_normal":
            in_channels = 12
        return in_channels

    def prepare_input(
        self,
        s_point: torch.Tensor,
        s_normal: torch.Tensor,
        t_point: torch.Tensor,
        t_normal: torch.Tensor,
        mode: str = "point_normal",
    ):
        if mode == "point_normal":  # (B, H, W, 12)
            x = torch.cat([s_point, s_normal, t_point, t_normal], dim=-1)
        elif mode == "point":
            x = torch.cat([s_point, t_point], dim=-1)  # (B, H, W, 6)
        elif mode == "normal":
            x = torch.cat([s_normal, t_normal], dim=-1)  # (B, H, W, 6)
        else:
            raise AttributeError(f"No {mode} that works.")
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        return x

    def compute_final_activation(
        self,
        x: torch.Tensor,
        mode: str = "elu",
        max_weight: float | None = None,
    ):
        assert mode in ["none", "relu", "elu", "exp", "one", "relu_one", "abs"]
        if mode == "elu":
            x = torch.nn.functional.elu(x) + 1.0
        elif mode == "one":
            x = x + 1.0
        elif mode == "relu":
            x = torch.relu(x)
        elif mode == "relu_one":
            x = torch.relu(x) + 1.0
        elif mode == "exp":
            x = torch.exp(x)
        elif mode == "abs":
            x = torch.abs(x)
        elif mode == "none":
            pass
        else:
            raise AttributeError(f"No {mode} that works.")
        if max_weight is not None:
            x = torch.min(x, torch.tensor(max_weight))
        return x


class CNNWeightModule(WeightModule):
    def __init__(
        self,
        features: int = 100,
        depth: int = 1,
        mode: str = "point",  # "point", "normal", "point_normal"
        max_weight: float | None = None,
        final_activation: str = "none",  # none, relu, elu, exp
        device: str = "cuda",
    ):
        super().__init__()
        self.max_weight = max_weight
        self.mode = mode
        self.final_activation = final_activation
        self.cnn = CNNMLP(
            in_channels=self.compute_in_channels(mode=mode),
            out_channels=1,
            hidden_channels=features,
            kernal_size=3,
            num_layers=depth,
        )
        self.to(device)

    def forward(
        self,
        s_point: torch.Tensor,
        s_normal: torch.Tensor,
        t_point: torch.Tensor,
        t_normal: torch.Tensor,
    ):
        x = self.prepare_input(s_point, s_normal, t_point, t_normal, self.mode)
        x = self.cnn(x)  # (B, C, H, W)
        x = self.compute_final_activation(x, self.final_activation, self.max_weight)
        return dict(weights=x, latent=None)


class UNetWeightModule(WeightModule):
    def __init__(
        self,
        features: int = 64,
        depth: int = 4,
        size: int = 256,
        mode: str = "point",  # "point", "normal", "point_normal"
        max_weight: float = 100.0,
        layer_norm: bool = True,
        final_activation: str = "none",  # none, relu, elu, exp
        device: str = "cuda",
    ):
        super().__init__()

        self.max_weight = max_weight
        self.mode = mode
        self.final_activation = final_activation
        self.init_features = features
        self.layer_norm = layer_norm
        self.depth = depth
        self.size = size
        in_channels = self.compute_in_channels(mode=mode)

        # Contracting Path (Encoder)
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        for i in range(depth):
            self.encoders.append(self._block(in_channels, features))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = features
            features *= 2
        # Bottleneck
        self.bottleneck = self._block(features // 2, features)
        # Expansive Path (Decoder)
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth):
            features //= 2
            upconv = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
            self.upconvs.append(upconv)
            self.decoders.append(self._block(features * 2, features))
        # Final Convolution
        self.conv = nn.Conv2d(in_channels=features, out_channels=1, kernel_size=1)
        self.to(device)

    def forward(
        self,
        s_point: torch.Tensor,
        s_normal: torch.Tensor,
        t_point: torch.Tensor,
        t_normal: torch.Tensor,
    ):
        # prepare_input
        x = self.prepare_input(s_point, s_normal, t_point, t_normal, self.mode)
        B, C, H, W = x.shape
        x = self._pad(x, height=x.shape[-2], width=x.shape[-1])

        # encode blocks
        encoders_output = []
        for i in range(self.depth):
            x = self.encoders[i](x)
            encoders_output.append(x)
            x = self.pools[i](x)

        # bottleneck layers
        bottleneck = self.bottleneck(x)  # (B, C, H, W)

        # decoder blocks
        for i in range(self.depth):
            x = self.upconvs[i](bottleneck if i == 0 else x)
            enc_output = encoders_output[-(i + 1)]
            x = torch.cat((x, enc_output), dim=1)
            x = self.decoders[i](x)

        # final block
        x = self.conv(x)
        x = self.compute_final_activation(x, self.final_activation, self.max_weight)

        # prepare output
        x = self._unpad(x, height=H, width=W)  # (B, 1, H, W)
        x = x.squeeze(1)  # (B, H, W)

        return dict(weights=x, latent=bottleneck)

    def _block(self, in_channels: int, features: int):
        # the size for the layer normalization
        scale = features // self.init_features
        size = self.size // scale
        # define the cnn block
        c1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=features,
            kernel_size=3,
            padding=1,
        )
        l1 = nn.LayerNorm([size, size])
        r1 = nn.ReLU(inplace=True)
        c2 = nn.Conv2d(
            in_channels=features,
            out_channels=features,
            kernel_size=3,
            padding=1,
        )
        l2 = nn.LayerNorm([size, size])
        r2 = nn.ReLU(inplace=True)
        if self.layer_norm:
            return nn.Sequential(c1, l1, r1, c2, l2, r2)
        return nn.Sequential(c1, r1, c2, r2)

    def _pad(self, x: torch.Tensor, height: int, width: int):
        # Desired output dimensions
        target_height = self.size
        target_width = self.size

        # Calculate padding for height and width
        pad_height = target_height - height
        pad_width = target_width - width

        # Pad equally on both sides
        padding = [
            pad_width // 2,
            pad_width - pad_width // 2,
            pad_height // 2,
            pad_height - pad_height // 2,
        ]  # (left, right, top, bottom)

        # Apply padding
        return F.pad(x, padding)

    def _unpad(self, x: torch.Tensor, height: int, width: int):
        # Desired output dimensions
        target_height = self.size
        target_width = self.size

        # Calculate padding for height and width
        pad_height = target_height - height
        pad_width = target_width - width

        # Slice back to the original shape (135, 240)
        start_height = pad_height // 2
        end_height = start_height + height

        start_width = pad_width // 2
        end_width = start_width + width

        return x[:, :, start_height:end_height, start_width:end_width]


class DummyWeightModule(nn.Module):
    def forward(self, s_point: torch.Tensor, **kwargs):
        B, H, W, _C = s_point.shape
        weights = torch.ones((B, H, W), device=s_point.device)
        return dict(weights=weights, latent=None)
