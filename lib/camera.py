import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_intrinsics(data_dir: str | Path):
    """The camera intrinsics for the kinect RGB-D sequence.

    For more information please refere to:
    https://github.com/zinsmatt/7-Scenes-Calibration
    https://cvg.cit.tum.de/data/datasets/rgbd-dataset/intrinsic_calibration

    Args:
        data_dir (str | Path): The root path of the dphm kinect dataset.

    Returns:
        The intrinsics for the kinect camera.
    """

    path = Path(data_dir) / "calibration.json"
    with open(path) as f:
        intrinsics = json.load(f)

    # just the focal lengths (fx, fy) and the optical centers (cx, cy)
    return {
        "fx": intrinsics["color"]["fx"],
        "fy": intrinsics["color"]["fy"],
        "cx": intrinsics["color"]["cx"],
        "cy": intrinsics["color"]["cy"],
    }


def camera2screen(
    points: Any,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
):
    """Converts points in camera coordinates to points in pixel coordinates.

    We follow the perspective projection described in the CV2 lecture chapter03
    image formation, with some modifications. We return a 3D vector of (N, 3) where
    the z-coordinate is the actual depth value from the camera coordinate. The x,y are
    in pixel coordinates. An example would be:

    Input:
    fx = 914.415
    fy = 914.03
    cx = 959.598
    cy = 547.202
    xyz_camera = [-0.051, -0.042,  0.575] (x, y, z_c)

    Output:
    uvz_pixel = [878.0, 480.0, 0.575] (u, v, z_c)

    Args:
        points (Any): The points in camera coordinate system .
        fx (float): The intrinsics of focal lengths.
        fy (float): The intrinsics of focal lengths.
        cx (float): The intrinsics of pixel coordinates of the camera optical center.
        cy (float): The intrinsics of pixel coordinates of the camera optical center.

    Returns:
        (np.ndarray | torch.Tensor): The points in pixel coordinate system.
    """
    # extract the coordinates
    xc = points[:, 0]
    yc = points[:, 1]
    zc = points[:, 2]

    # from camera space to screen space
    xs = xc / zc
    ys = yc / zc

    # from screen space to pixxel space
    u = cx + fx * xs
    v = cy + fy * ys
    depth = zc

    if isinstance(u, torch.Tensor):
        return torch.stack([u, v, depth], dim=-1)  # (N, 3)
    return np.stack([u, v, depth], axis=-1)  # (N, 3)
