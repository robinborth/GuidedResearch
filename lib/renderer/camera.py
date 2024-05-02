import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torchvision.transforms import v2


def load_intrinsics(data_dir: str | Path, return_tensor: str = "dict"):
    """The camera intrinsics for the kinect RGB-D sequence.

    For more information please refere to:
    https://github.com/zinsmatt/7-Scenes-Calibration
    https://cvg.cit.tum.de/data/datasets/rgbd-dataset/intrinsic_calibration

    Args:
        data_dir (str | Path): The root path of the dphm kinect dataset.

    Returns:
        The intrinsics for the kinect camera.
    """
    assert return_tensor in ["dict", "pt"]

    path = Path(data_dir) / "calibration.json"
    with open(path) as f:
        intrinsics = json.load(f)

    # just the focal lengths (fx, fy) and the optical centers (cx, cy)
    K = {
        "fx": intrinsics["color"]["fx"],
        "fy": intrinsics["color"]["fy"],
        "cx": intrinsics["color"]["cx"],
        "cy": intrinsics["color"]["cy"],
    }

    if return_tensor == "pt":
        return torch.tensor(
            [[K["fx"], 0.0, K["cx"]], [0.0, K["fy"], K["cy"]], [0.0, 0.0, 1.0]]
        )

    return K


def camera2pixel(
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

    # from camera space to pixel space
    u = cx + fx * (xc / zc)
    v = cy + fy * (yc / zc)
    depth = zc

    if isinstance(u, torch.Tensor):
        return torch.stack([u, v, depth], dim=-1)  # (N, 3)
    return np.stack([u, v, depth], axis=-1)  # (N, 3)


def pixel2camera(
    points: Any,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
):
    """Converts points in pixel coordinates to points in camera coordinates.

    We follow the intrinsic camera calibration on the kinect documentation of the CV2
    chair. However we asume that the points that are in pixel coordinate system, e.g.
    have coordinates (u,v) with an additional depth information. Hence the dimension is
    (N, 3), where we have a 1:1 correspondence between pixels and their depth. For more
    information refer to:

    https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats


    Args:
        points (Any): The points in pixel space.
        fx (float): The intrinsics of focal lengths.
        fy (float): The intrinsics of focal lengths.
        cx (float): The intrinsics of pixel coordinates of the camera optical center.
        cy (float): The intrinsics of pixel coordinates of the camera optical center.

    Returns:
        (np.ndarray | torch.Tensor): The points in camera coordinate system.
    """
    # extract the coordinates
    u = points[:, 0]
    v = points[:, 1]
    zc = points[:, 2]

    # convert pixel to camera coordinates
    xc = (u - cx) * (zc / fx)
    yc = (v - cy) * (zc / fy)

    if isinstance(u, torch.Tensor):
        return torch.stack([xc, yc, zc], dim=-1)  # (N, 3)
    return np.stack([xc, yc, zc], axis=-1)  # (N, 3)


def depth2camera(
    depth: torch.Tensor,
    K: torch.Tensor,
    scale: float = 1.0,
):
    """Converts depth image in camera coordinates as image.

    After this transformation each pixel cooresponds to the 3d coordinate in camera
    coordinate system. Further we can specify the scale of the output image, hence
    reduce the image size, which is usefull for the coarse to fine strategy. Note that
    we need to make sure to interpolate the binary mask of backgfound and foreground
    properly.

    Args:
        depth (torch.Tensor): The (H,W) depth image, where each pixel stores the depth
            value in camera coordinate measurements.
        K (torch.Tensor): The 3x3 intrinsic matrix.
        scale (float): The scale of the output camera

    Returns:
        (torch.Tensor): The image of the 3d camera coordinates, where values of 0 depth
        e.g. z dim cooresponds to background.
    """
    H, W = int(depth.shape[0] * scale), int(depth.shape[1] * scale)

    # modify the output size of the camera
    K = K.clone()
    K[0, 0] *= scale  # fx
    K[1, 1] *= scale  # fy
    K[0, 2] *= scale  # cx
    K[1, 2] *= scale  # cy

    # calc the mask with the backgound and then output the forground, e.g the resize is
    # downing the interpolation in a way that by boolen we just keep true.
    b_mask = v2.functional.resize((depth == 0.0).unsqueeze(0), size=(H, W)).squeeze(0)

    # get the new size of the depth image
    depth = v2.functional.resize(depth.unsqueeze(0), size=(H, W)).squeeze(0)

    # span the pixel indexes
    x = torch.arange(W)
    y = torch.arange(H)
    idx = torch.stack(torch.meshgrid(y, x), dim=-1).flip(-1)

    # get the points in camera coordinates, but with the new resolution
    points = torch.concat([idx, depth.unsqueeze(-1)], dim=-1)
    points[:, :, 0] *= points[:, :, 2]
    points[:, :, 1] *= points[:, :, 2]
    camera = K.inverse() @ points.permute(2, 0, 1).reshape(3, -1)
    camera = camera.reshape(3, points.shape[0], points.shape[1]).permute(1, 2, 0)

    # the background is just zero
    camera[b_mask, :] = 0.0
    return camera
