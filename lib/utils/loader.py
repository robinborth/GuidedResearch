import json
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lib.utils.image import biliteral_filter

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


def convert_dict_from_np(np_dict: dict, return_tensors: str = "pt"):
    assert return_tensors in ["pt", "np"]
    if return_tensors == "pt":
        for key, value in np_dict.items():
            if isinstance(value, np.ndarray):
                np_dict[key] = torch.tensor(value)
    return np_dict


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
):
    """Load the RGB data for the kinect dataset.

    Args:
        data_dir (str | Path): The root path of the dphm kinect dataset.
        idx (int): The sequence index of the image of the recording, e.g. the dataset.
        return_tensor: (str): The return type of the image, either "img", "np" or "pt",
            where img is the PIL.Image format.
    """
    assert return_tensor in ["img", "np", "pt"]
    mask = load_mask(data_dir=data_dir, idx=idx, return_tensor="np")

    path = Path(data_dir) / "color" / f"{idx:05}.png"
    color = np.asarray(Image.open(path)).copy()
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
# FLAME
########################################################################################


def load_flame(
    flame_dir: str | Path,
    model_name: str = "flame.pkl",
    return_tensors: str = "pt",
):
    """Loads the FLAME model.

    https://github.com/soubhiksanyal/FLAME_PyTorch/blob/master/requirements.txt
    the requirenemnts needs to be fixed and we need to have chumpy installed
    also python 3.10 is required with an "older" numpy version

    Args:
        flame_dir (str | Path): The root directory of the flame model.
        model_name (str, optional): The flame model. Defaults to "flame2023.pkl".
        return_tensors (str, optional): Describes which format the flame model will be
            return, similar to huggingface, e.g. "pt" or "np". Defaults to "pt".

    Returns:
        dict: The FLAME model as dictionary. For more informations please refere to the
            paper or the FLAME implementations.
    """

    # load the model, see description why we load it like that.
    # warnings.filterwarnings("ignore", category=DeprecationWarning)
    path = Path(flame_dir) / model_name
    with open(path, "rb") as f:
        flame = pickle.load(f, encoding="latin1")

    # convert to proper numpy
    bs_style = str(flame["bs_style"])
    bs_type = str(flame["bs_type"])

    faces = np.array(flame["f"], dtype=np.int64)
    v_template = np.array(flame["v_template"], dtype=np.float32)

    shapedirs = np.array(flame["shapedirs"], dtype=np.float32)
    posedirs = np.array(flame["posedirs"], dtype=np.float32)
    weights = np.array(flame["weights"], dtype=np.float32)

    J_regressor = np.array(flame["J_regressor"].todense(), dtype=np.float32)
    J = np.array(flame["J"], dtype=np.float32)
    kintree_table = np.array(flame["kintree_table"], dtype=np.int64)

    # group the flame model together
    flame_dict = {
        "bs_style": bs_style,
        "bs_type": bs_type,
        "f": faces,
        "v_template": v_template,
        "shapedirs": shapedirs,
        "posedirs": posedirs,
        "weights": weights,
        "J_regressor": J_regressor,
        "J": J,
        "kintree_table": kintree_table,
    }

    # convert the numpy arrays to torch tensors
    return convert_dict_from_np(flame_dict, return_tensors=return_tensors)


def load_static_landmark_embedding(flame_dir: str | Path, return_tensors: str = "np"):
    path = Path(flame_dir) / "mediapipe_landmark_embedding.npz"
    lm = np.load(path)
    landmark_dict = {
        "lm_face_idx": lm["lmk_face_idx"].astype(np.int64),
        "lm_bary_coords": lm["lmk_b_coords"].astype(np.float32),
        "lm_mediapipe_idx": lm["landmark_indices"].astype(np.int64),
    }
    return convert_dict_from_np(landmark_dict, return_tensors=return_tensors)


def load_flame_masks(flame_dir: str | Path, return_tensors: str = "np"):
    path = Path(flame_dir) / "FLAME_masks.pkl"
    with open(path, "rb") as f:
        flame_masks = pickle.load(f, encoding="latin1")
    for key in flame_masks.keys():
        flame_masks[key] = flame_masks[key].astype(np.int64)
    flame_masks["full"] = np.arange(5023, dtype=np.int64)
    return convert_dict_from_np(flame_masks, return_tensors=return_tensors)


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
