import pickle
from pathlib import Path

import numpy as np
import torch

from lib.data.utils import convert_dict_from_np


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
    return convert_dict_from_np(flame_masks, return_tensors=return_tensors)


def flame_faces_mask(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    vertices_mask: torch.Tensor | None = None,
):
    """Calculates the triangular faces mask based on the masked vertices.

    Args:
        vertices (torch.Tensor): The vertices in camera coordinate system (V, 3)
        faces (torch.Tensor): The indexes of the vertices, e.g. the faces (F, 3)
        vertices_mask (torch.Tensor): The idx of the vertices that should be
            included in the rendering and computation.

    Returns:
        (torch.Tensor): A boolean mask of the faces that should be used for the
            computation, the final dimension of the mask is (F, 3).
    """
    if vertices_mask is None:
        vertices_mask = torch.arange(vertices.shape[0], device=vertices.device)
    vertices_mask = vertices_mask.expand(*faces.shape, -1).to(vertices.device)
    return (faces.unsqueeze(-1) == vertices_mask).any(dim=-1).all(dim=-1)


def bary_coord_interpolation(
    faces: torch.Tensor,
    attributes: torch.Tensor,
    bary_coords: torch.Tensor,
):
    """Rendering of the attributes with mesh rasterization.

    Args:
        faces (torch.Tensor): The indexes of the vertices, e.g. the faces (F, 3)
        attributes (torch.Tensor): The attributes per vertex of dim (V, D)
        bary_coords (torch.Tensor): The interpolation coeficients of dim (F, 3)

    Returns:
        (torch.Tensor): Vertices with the attributes barycentric interpolated (F, D)
    """
    vertex_attribute = attributes[faces]  # (H, W, 3, D)
    return (bary_coords.unsqueeze(-1) * vertex_attribute).sum(-2)
