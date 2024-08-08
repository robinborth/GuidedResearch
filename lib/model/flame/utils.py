import pickle
from pathlib import Path

import numpy as np
import torch

########################################################################################
# Utils
########################################################################################


def convert_dict_from_np(np_dict: dict, return_tensors: str = "pt"):
    assert return_tensors in ["pt", "np"]
    if return_tensors == "pt":
        for key, value in np_dict.items():
            if isinstance(value, np.ndarray):
                np_dict[key] = torch.tensor(value)
    return np_dict


########################################################################################
# Flame
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
