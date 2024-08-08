import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lib.data.preprocessing import biliteral_filter

########################################################################################
# Utils
########################################################################################


def convert_tensor_from_np(X: np.ndarray, return_tensor: str = "np"):
    assert return_tensor in ["img", "np", "pt"]
    if return_tensor == "img":
        return Image.fromarray(X.astype(np.uint8))
    if return_tensor == "pt":
        return torch.tensor(X, dtype=torch.float32)
    return X.astype(dtype=np.float32)


########################################################################################
# DPHM: RGB and Depth
########################################################################################


def load_mask(
    data_dir: str | Path,
    idx: int,
    return_tensor: str = "np",
    threshold: float = 200.0,
):
    """Creates a binary mask for the kinect images.

    The creation of the mask is based on the depth observations from the dataset and the
    already filtered image from the DPHMs dataset, with some threshold.

    Args:
        data_dir (str | Path): The path of the dphm dataset.
        idx (int): The index of the sequence image.
        return_tensor: (str): The return type of the image, either "np" or "pt".

    Returns: The binary mask, where True referes to the foreground, e.g. the face
        and False to the background.
    """
    assert return_tensor in ["pt", "np"]

    path = Path(data_dir) / "depth_normals_bilateral" / f"{idx:05}_depth.jpg"
    mask_image = Image.open(path)

    mask = np.asarray(mask_image).mean(-1) < threshold
    if return_tensor == "pt":
        return torch.tensor(mask)
    return mask


def load_color(
    data_dir: str | Path,
    idx: int,
    value: float | int = 0,
    return_tensor: str = "img",
    mask: bool = False,
):
    """Load the RGB data for the kinect dataset.

    Args:
        data_dir (str | Path): The root path of the dphm kinect dataset.
        idx (int): The sequence index of the image of the recording, e.g. the dataset.
        return_tensor: (str): The return type of the image, either "img", "np" or "pt",
            where img is the PIL.Image format.
    """
    assert return_tensor in ["img", "np", "pt"]

    path = Path(data_dir) / "color" / f"{idx:05}.png"
    color = np.asarray(Image.open(path)).copy()

    if mask:
        mask = load_mask(data_dir=data_dir, idx=idx, return_tensor="np")
        color[~mask] = value

    return convert_tensor_from_np(color, return_tensor=return_tensor)


def load_depth(
    data_dir: str | Path,
    idx: int,
    value: float | int = 0,
    depth_factor: float = 1000,
    return_tensor: str = "np",
    smooth: bool = False,
):
    """Load the depth data for the kinect dataset.

    The depth images are scaled by a factor of 1000, i.e., a pixel value of 1000 in
    the depth image corresponds to a distance of 1 meter from the camera. A pixel value
    of 0 means missing value/no data.

    For more information please refere to:
    https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats

    Args:
        data_dir (str | Path): The root path of the dphm kinect dataset.
        idx (int): The sequence index of the image of the recording, e.g. the dataset.
        depth_factor: (float): The pixel to depth ratio. i.e., a pixel value of 5000 in
            the depth image corresponds to a distance of 1 meter from the camera.
        return_tensor: (str): The return type of the image, either "np" or "pt".

    Returns: The depth overservations in m.
    """
    assert return_tensor in ["np", "pt"]
    mask = load_mask(data_dir=data_dir, idx=idx, return_tensor="np")

    path = Path(data_dir) / "depth" / f"{idx:05}.png"
    depth_image = np.asarray(Image.open(path)).copy()

    depth = (depth_image / depth_factor).astype(np.float32)
    depth[~mask] = depth[mask].max()

    if smooth:
        depth = biliteral_filter(
            image=depth,
            dilation=30,
            sigma_color=150,
            sigma_space=150,
        )
    depth[~mask] = value

    return convert_tensor_from_np(depth, return_tensor=return_tensor)


def load_normal(
    data_dir: str | Path,
    idx: int,
    value: float | int = 0,
    return_tensor: str = "np",
    smooth: bool = False,
):
    """Load the normal data for the kinect dataset.

    Args:
        data_dir (str | Path): The root path of the dphm kinect dataset.
        idx (int): The sequence index of the image of the recording, e.g. the dataset.
        return_tensor: (str): The return type of the image, either "np" or "pt".
        smooth (bool): Apply the bilateral filter.

    Returns: The normal ranging between (-1, 1).
    """

    assert return_tensor in ["np", "pt"]
    mask = load_mask(data_dir=data_dir, idx=idx, return_tensor="np")

    path = Path(data_dir) / "depth_normals_bilateral" / f"{idx:05}_normal.jpg"
    normal_image = np.asarray(Image.open(path)).copy()

    if smooth:
        normal_image = biliteral_filter(
            image=normal_image,
            dilation=30,
            sigma_color=250,
            sigma_space=250,
        )

    normal = ((normal_image / 255) * 2) - 1
    normal = normal / np.linalg.norm(normal, axis=-1)[..., None]
    normal[~mask] = value

    return convert_tensor_from_np(normal, return_tensor=return_tensor)


########################################################################################
# DPHMs Landmarks
########################################################################################


def load_pipnet_image(data_dir: str | Path, idx: int) -> Image.Image:
    path = Path(data_dir) / "color/PIPnet_annotated_images" / f"{idx:05}.png"
    return Image.open(path)


def load_pipnet_landmark_2d(
    data_dir: str | Path,
    idx: int,
    return_tensor: str = "np",
) -> np.ndarray | torch.Tensor:
    assert return_tensor in ["np", "pt"]
    path = Path(data_dir) / "color/PIPnet_landmarks" / f"{idx:05}.npy"
    lm = np.load(path)
    return convert_tensor_from_np(lm, return_tensor=return_tensor)


def load_pipnet_landmark_3d(
    data_dir: str | Path,
    idx: int,
    return_tensor: str = "np",
) -> np.ndarray | torch.Tensor:
    assert return_tensor in ["np", "pt"]
    path = Path(data_dir) / "lms_3d_pip_new" / f"{idx:05}.npy"
    lm = np.load(path)
    return convert_tensor_from_np(lm, return_tensor=return_tensor)


def load_mediapipe_image(data_dir: str | Path, idx: int) -> Image.Image:
    path = Path(data_dir) / "color/Mediapipe_annotated_images" / f"{idx:05}.png"
    return Image.open(path)


def load_mediapipe_landmark_2d(
    data_dir: str | Path,
    idx: int,
    return_tensor: str = "np",
) -> np.ndarray | torch.Tensor:
    assert return_tensor in ["np", "pt"]
    path = Path(data_dir) / "color/Mediapipe_landmarks" / f"{idx:05}.npy"
    lm = np.load(path)
    return convert_tensor_from_np(lm, return_tensor=return_tensor)


def load_mediapipe_landmark_3d(
    data_dir: str | Path,
    idx: int,
    return_tensor: str = "np",
) -> np.ndarray | torch.Tensor:
    assert return_tensor in ["np", "pt"]
    path = Path(data_dir) / "lms_3d_mp_new" / f"{idx:05}.npy"
    lm = np.load(path)
    return convert_tensor_from_np(lm, return_tensor=return_tensor)


########################################################################################
# DPHMs Normals and Points in 3D
########################################################################################


def load_normals_3d(
    data_dir: str | Path,
    idx: int,
    return_tensor: str = "np",
) -> np.ndarray | torch.Tensor:
    assert return_tensor in ["np", "pt"]
    path = Path(data_dir) / "normals_new_maskmouth" / f"{idx:05}.npy"
    normals = np.load(path)
    return convert_tensor_from_np(normals, return_tensor=return_tensor)


def load_points_3d(
    data_dir: str | Path,
    idx: int,
    return_tensor: str = "np",
) -> np.ndarray | torch.Tensor:
    assert return_tensor in ["np", "pt"]
    path = Path(data_dir) / "points_new_maskmouth" / f"{idx:05}.npy"
    points = np.load(path)
    return convert_tensor_from_np(points, return_tensor=return_tensor)


########################################################################################
# DPHM Intrinsics
########################################################################################


def load_intrinsics(
    data_dir: str | Path,
    return_tensor: str = "dict",
):
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
            [
                [K["fx"], 0.0, K["cx"]],
                [0.0, K["fy"], K["cy"]],
                [0.0, 0.0, 1.0],
            ]
        )
    return K
