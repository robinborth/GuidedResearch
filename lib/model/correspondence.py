import torch
import torch.nn as nn

from lib.model.common import CNNMLP


class OpticalFlowCorrespondenceModule(nn.Module):
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
            out_channels=2,
            hidden_channels=hidden_channels,
            kernal_size=kernal_size,
            num_layers=num_layers,
        )
        self.to(device)

    def create_grid(self, s_delta: torch.Tensor):
        B, H, W, _ = s_delta.shape
        # create the pixel grid
        x = torch.linspace(-1, 1, steps=W, device=s_delta.device)
        y = torch.linspace(-1, 1, steps=H, device=s_delta.device)
        y, x = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack([y, x], dim=-1).expand(B, H, W, 2)
        # add the optical flow delta between (-1, 1)
        grid = grid + s_delta
        return grid

    def sample_grid(self, value: torch.Tensor, grid: torch.Tensor):
        B, H, W, C = value.shape

        # scale grid values from [-1, 1] to [0, W-1] for x and [0, H-1] for y
        x = ((grid[..., 0] + 1) * (W - 1)) / 2
        y = ((grid[..., 1] + 1) * (H - 1)) / 2

        # get the integer part of the coordinates
        x0 = torch.floor(x).long()
        x1 = x0 + 1
        y0 = torch.floor(y).long()
        y1 = y0 + 1

        # create masks for values that need clamping
        x0_clamp_mask = (x0 < 0) | (x0 >= W)
        x1_clamp_mask = (x1 < 0) | (x1 >= W)
        y0_clamp_mask = (y0 < 0) | (y0 >= H)
        y1_clamp_mask = (y1 < 0) | (y1 >= H)
        clamp_mask = x0_clamp_mask | x1_clamp_mask | y0_clamp_mask | y1_clamp_mask

        # clamp the values
        x0 = torch.clamp(x0, 0, W - 1)  # (B, H, W)
        x1 = torch.clamp(x1, 0, W - 1)  # (B, H, W)
        y0 = torch.clamp(y0, 0, H - 1)  # (B, H, W)
        y1 = torch.clamp(y1, 0, H - 1)  # (B, H, W)

        # gather pixel values at the corners
        batch_indices = torch.arange(B, device=value.device).view(-1, 1, 1)
        Ia = value[batch_indices, y0, x0, :]  # top-left
        Ib = value[batch_indices, y1, x0, :]  # bottom-left
        Ic = value[batch_indices, y0, x1, :]  # top-right
        Id = value[batch_indices, y1, x1, :]  # bottom-right

        # get the fractional part of the coordinates
        wa = (x1.float() - x) * (y1.float() - y)
        wb = (x1.float() - x) * (y - y0.float())
        wc = (x - x0.float()) * (y1.float() - y)
        wd = (x - x0.float()) * (y - y0.float())

        # bilinear interpolate the cornes
        interpolated_values = (
            wa.unsqueeze(-1) * Ia
            + wb.unsqueeze(-1) * Ib
            + wc.unsqueeze(-1) * Ic
            + wd.unsqueeze(-1) * Id
        )

        # zero out the where we need to clamp
        zero_tensor = torch.zeros_like(interpolated_values)
        interpolated_values[clamp_mask] = zero_tensor[clamp_mask]

        return interpolated_values

    def predict(self, s_point: torch.Tensor, t_point: torch.Tensor, **kwargs):
        x = torch.cat([s_point, t_point], dim=-1)  # (B, H, W, 6)
        x = x.permute(0, 3, 1, 2)  # (B, 6, H, W)
        s_delta = self.cnn(x)  # (B, W, H)
        s_delta = s_delta.permute(0, 2, 3, 1)  # (B, W, H, 2)
        return {"s_delta": s_delta}

    def transform(self, s_delta: torch.Tensor, t_value: torch.Tensor):
        grid = self.create_grid(s_delta=s_delta)
        samples = self.sample_grid(grid=grid, value=t_value)
        return samples

    def mask(self, s_mask: torch.Tensor, t_mask: torch.Tensor, s_delta: torch.Tensor):
        # compute the grid with the delta added
        grid = self.create_grid(s_delta=s_delta)
        # the values that are not in the mask needs to be marked diry
        value = t_mask.clone().float().unsqueeze(-1)
        # sample the grid
        samples = self.sample_grid(grid=grid, value=value)
        # infere the mask from the dirty nans
        t_new_mask = samples.bool().squeeze(-1)
        # compute the mask, where the source points found a correspondence point
        mask = t_new_mask & s_mask
        return mask


class ProjectiveCorrespondenceModule(nn.Module):
    def __init__(self, d_threshold: float = 0.1, n_threshold: float = 0.9):
        self.d_threshold = d_threshold
        self.n_threshold = n_threshold

    def predict(
        self,
        s_point: torch.Tensor,
        s_normal: torch.Tensor,
        t_point: torch.Tensor,
        t_normal: torch.Tensor,
    ):
        return {
            "s_point": s_point,
            "s_normal": s_normal,
            "t_point": t_point,
            "t_normal": t_normal,
        }

    def mask(
        self,
        s_mask: torch.Tensor,
        s_point: torch.Tensor,
        s_normal: torch.Tensor,
        t_mask: torch.Tensor,
        t_point: torch.Tensor,
        t_normal: torch.Tensor,
    ):
        # per pixel distance in 3d, just the length
        dist = torch.norm(t_point - s_point, dim=-1)  # (B, W, H)
        # calculate the forground mask
        f_mask = s_mask & t_mask  # (B, W, H)
        # the depth mask based on some epsilon of distance 10cm
        d_mask = dist < self.d_threshold  # (B, W, H)
        # dot product, e.g. coresponds to an angle
        normal_dot = (s_normal * t_normal).sum(-1)
        n_mask = normal_dot > self.n_threshold  # (B, W, H)
        # final loss mask of silhouette, depth and normal threshold
        final_mask = d_mask & f_mask & n_mask
        assert final_mask.sum()  # we have some overlap
        return final_mask

    def transform(self, t_value: torch.Tensor, **kwargs):
        return t_value
