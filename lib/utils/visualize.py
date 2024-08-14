import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from matplotlib import cm

from lib.model import Flame
from lib.renderer import Renderer
from lib.renderer.camera import Camera
from lib.utils.distance import (
    landmark_2d_distance,
    landmark_3d_distance,
    point2plane_distance,
    point2point_distance,
    regularization_distance,
)


def change_color(color: torch.Tensor, mask: torch.Tensor, code: int = 0):
    img = color.clone()
    img[..., code] = 0
    img[~mask] = 255
    return img


def load_pcd(points, color=None):
    if color is None:
        color = [255, 0, 0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    color = [np.array(color, dtype=np.uint8)] * points.shape[0]
    pcd.colors = o3d.utility.Vector3dVector(np.stack(color))
    return pcd


def visualize_grid(images: torch.Tensor, figsize: int = 15):
    # visualize as grid
    images = images.detach().cpu().numpy()
    batch_size = images.shape[0]
    grid_size = int(torch.ceil(torch.sqrt(torch.tensor(batch_size))))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(figsize, figsize))

    # Ensure axes is always an array
    if grid_size == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i in range(grid_size * grid_size):
        ax = axes[i]
        if i < batch_size:
            ax.imshow(images[i])
        ax.axis("off")
    # show with matplotlib
    plt.show()


def visualize_params(
    flame: Flame,
    renderer: Renderer,
    params: dict[str, torch.Tensor],
    color: int = 0,
):
    assert color in [0, 1, 2]
    out = flame.render(renderer=renderer, params=params)
    img = change_color(color=out["color"], mask=out["mask"], code=color)
    visualize_grid(img, figsize=15)


def visualize_point2point_error(
    s_point: torch.Tensor,
    t_point: torch.Tensor,
    t_mask: torch.Tensor,
    max_error: float = 1e-02,
):
    norm = plt.Normalize(0.0, vmax=max_error)
    cmap = plt.get_cmap("jet")
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    loss = point2point_distance(s_point, t_point)  # (B, W, H)
    error_map = loss.detach().cpu().numpy()  # dist in m
    error_map = torch.from_numpy(sm.to_rgba(error_map)[:, :, :3])
    error_map[~t_mask, :] = 1.0
    error_map = (error_map * 255).to(torch.uint8)
    return error_map


def visualize_point2plane_error(
    s_point: torch.Tensor,
    t_point: torch.Tensor,
    t_normal: torch.Tensor,
    t_mask: torch.Tensor,
    max_error: float = 1e-02,
):
    norm = plt.Normalize(0.0, vmax=max_error)
    cmap = plt.get_cmap("jet")
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    loss = point2plane_distance(p=s_point, q=t_point, n=t_normal)  # (B, W, H)
    error_map = loss.detach().cpu().numpy()  # dist in m
    error_map = torch.from_numpy(sm.to_rgba(error_map)[:, :, :3])
    error_map[~t_mask, :] = 1.0
    error_map = (error_map * 255).to(torch.uint8)
    return error_map


def visualize_normal_error(
    s_normal: torch.Tensor,
    t_normal: torch.Tensor,
    t_mask: torch.Tensor,
    max_error: float = 1e-02,
):
    norm = plt.Normalize(0.0, vmax=max_error)
    cmap = plt.get_cmap("jet")
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    normal_dot = (s_normal * t_normal).sum(-1)
    normal_map = normal_dot.detach().cpu().numpy()
    normal_map = (1 - ((normal_map + 1) / 2)) * max_error
    normal_map = torch.from_numpy(sm.to_rgba(normal_map)[:, :, :3])
    normal_map[~t_mask, :] = 1.0
    normal_map = (normal_map * 255).to(torch.uint8)
    return normal_map


def visualize_depth_merged(
    s_mask: torch.Tensor,
    s_point: torch.Tensor,
    s_color: torch.Tensor,
    t_mask: torch.Tensor,
    t_point: torch.Tensor,
    t_color: torch.Tensor,
):
    img = s_color.clone()
    depth_mask = t_point[:, :, :, 2] < s_point[:, :, :, 2]
    color_mask = (depth_mask | (t_mask & ~s_mask)) & t_mask
    img[color_mask] = t_color[color_mask]
    return img


def visualize_merged(
    s_color: torch.Tensor,
    t_color: torch.Tensor,
    t_mask: torch.Tensor,
):
    img = s_color.clone()
    img[t_mask] = t_color[t_mask]
    return img


def visualize_landmark(
    color: torch.Tensor,
    lm_2d_ndc: torch.Tensor,
    renderer: Renderer,
):
    lm_2d_screen = renderer.camera.xy_ndc_to_screen(lm_2d_ndc)
    x_idx = lm_2d_screen[:, :, 0].to(torch.int32)
    y_idx = lm_2d_screen[:, :, 1].to(torch.int32)
    img = color.clone()
    img[y_idx, x_idx, :] = 255
    return img


def visualize_depth(point: torch.Tensor, renderer: Renderer):
    depth = renderer.point_to_depth(point)
    return renderer.depth_to_depth_image(depth)


def visualize_normal(normal: torch.Tensor, mask: torch.Tensor, renderer: Renderer):
    return renderer.normal_to_normal_image(normal, mask)
