import pickle
import warnings
from pathlib import Path

import numpy as np
import torch


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
    assert return_tensors in ["pt", "np"]
    if return_tensors == "pt":
        for key, value in flame_dict.items():
            if isinstance(value, np.ndarray):
                flame_dict[key] = torch.tensor(value)

    return flame_dict


def load_static_landmark_embedding(flame_dir: str | Path):
    path = Path(flame_dir) / "mediapipe_landmark_embedding.npz"
    return np.load(path)


def load_flame_masks(flame_dir: str | Path):
    path = Path(flame_dir) / "FLAME_masks.pkl"
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")
