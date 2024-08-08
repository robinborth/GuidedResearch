import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch

from lib.model import Flame
from lib.renderer import Renderer


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
    m_out = flame(**params)
    r_out = renderer.render_full(m_out["vertices"], faces=flame.faces)
    img = r_out["color"]
    img[..., color] = 0
    img[~r_out["mask"]] = 255
    visualize_grid(img, figsize=15)
