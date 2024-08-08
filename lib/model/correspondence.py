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

    def sample_grid(self, grid: torch.Tensor, value: torch.Tensor):
        value = value.permute(0, 3, 1, 2)
        samples = torch.nn.functional.grid_sample(
            input=value,
            grid=grid,
            mode="bilinear",
            align_corners=False,
            padding_mode="zeros",
        )
        samples = samples.permute(0, 2, 3, 1)
        return samples

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
