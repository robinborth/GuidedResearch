import lightning as L
import numpy as np
import torch
import torch.nn as nn

from lib.model.flame.lbs import lbs
from lib.model.flame.utils import (
    load_flame,
    load_flame_masks,
    load_static_landmark_embedding,
)
from lib.renderer import Camera, Rasterizer, Renderer
from lib.tracker.timer import TimeTracker


class Flame(L.LightningModule):
    def __init__(
        self,
        flame_dir: str = "/flame",
        shape_params: int = 100,
        expression_params: int = 50,
        vertices_mask: str = "face",  # full, face
        device: str = "cuda",
    ):
        super().__init__()

        # load the face model
        flame_model = load_flame(flame_dir=flame_dir, return_tensors="pt")

        # load the faces
        full_faces = flame_model["f"]
        self.full_faces = nn.Parameter(full_faces, requires_grad=False)  # (9976,3)
        face_mask = load_flame_masks(flame_dir, return_tensors="pt")[vertices_mask]
        masked_faces = self.mask_faces(flame_model["f"], face_mask)
        self.faces = torch.nn.Parameter(masked_faces, requires_grad=False)

        # load the mean vertices and pca bases
        self.v_template = nn.Parameter(flame_model["v_template"])  # (5023, 3)
        self.shapedirs = nn.Parameter(flame_model["shapedirs"])  # (5023, 3, 400)

        # linear blend skinning
        self.J_regressor = nn.Parameter(flame_model["J_regressor"])  # (5, 5023)
        self.lbs_weights = nn.Parameter(flame_model["weights"])  # (5023, 5)
        num_pose_basis = flame_model["posedirs"].shape[-1]  # (5023, 3, 36)
        posedirs = flame_model["posedirs"].reshape(-1, num_pose_basis).T
        self.posedirs = nn.Parameter(posedirs)  # (36, 5023 * 3)
        # indices of parents for each joints
        parents = flame_model["kintree_table"]  # (2, 5)
        parents[0, 0] = -1  # [-1, 0, 1, 1, 1]
        self.parents = nn.Parameter(parents[0], requires_grad=False)  # (5,)

        # flame masks, for different vertex idx
        face_mask = load_flame_masks(flame_dir, return_tensors="pt")[vertices_mask]
        self.vertices_mask = torch.nn.Parameter(face_mask, requires_grad=False)

        # flame landmarks to guide sparse landmark loss
        lms = load_static_landmark_embedding(flame_dir, return_tensors="pt")
        lm_faces = self.faces[lms["lm_face_idx"]]  # (105, )
        self.lm_faces = torch.nn.Parameter(lm_faces, requires_grad=False)
        lm_bary_coords = lms["lm_bary_coords"]  # (105, 3)
        self.lm_bary_coords = torch.nn.Parameter(lm_bary_coords, requires_grad=False)
        lm_idx = lms["lm_mediapipe_idx"]  # (105,)
        self.lm_mediapipe_idx = torch.nn.Parameter(lm_idx, requires_grad=False)

        # set default optimization parameters
        self.n_shape_params = shape_params
        self.n_expression_params = expression_params
        self.zero_shape = nn.Parameter(torch.zeros(300 - shape_params))
        self.zero_expression = nn.Parameter(torch.zeros(100 - expression_params))
        self.params = nn.ParameterDict(self.generate_default_params())

        self.global_params = ["shape_params", "scale"]
        self.local_params = [
            "expression_params",
            "global_pose",
            "transl",
            "neck_pose",
            "jaw_pose",
            "eye_pose",
        ]

        # move to cuda
        self.to(device=device)

    ####################################################################################
    # Core
    ####################################################################################

    def forward(
        self,
        shape_params=None,
        expression_params=None,
        global_pose=None,
        neck_pose=None,
        jaw_pose=None,
        eye_pose=None,
        transl=None,
        scale=None,
    ):
        """The forward pass the FLAME model.

        The vertices of the FLAME model are in world space. Hence the translation and
        the scaling are in world space.

        Args:
            shape_params (torch.tensor): (B, S') where S' is the number that was
                in the initialization and stores shape_params
            expression_params (torch.tensor): (B, E') where E' is the number that was
                in the initialization and stores expression_params
            global_pose (torch.tensor): number of global pose parameters (B, 3)
            neck_pose (torch.tensor): number of neck pose parameters (B, 3)
            jaw_pose (torch.tensor): number of jaw pose parameters (B, 3)
            eye_pose (torch.tensor): number of eye pose parameters (B, 6)
            transl(torch.tensor): number of translation parameters (B, 3)
            scale(torch.tensor): number of scale parameters (B, 1)

        Return:
            (torch.Tensor): The mesh vertices of dim (B, V, 3)
        """
        params = dict(
            shape_params=shape_params,
            expression_params=expression_params,
            global_pose=global_pose,
            neck_pose=neck_pose,
            jaw_pose=jaw_pose,
            eye_pose=eye_pose,
            transl=transl,
            scale=scale,
        )
        # fill the default values and determine the batch size
        B = 1
        for p_name, param in params.items():
            if param is None:
                params[p_name] = self.params[p_name]
            elif param.dim() == 2:  # override the batch_size
                B = max(param.shape[0], B)

        # check that the shape and expression params are correct
        if params["shape_params"].shape[-1] != self.n_shape_params:
            raise ValueError(f"Size of shape_params: {self.n_shape_params}")
        if params["expression_params"].shape[-1] != self.n_expression_params:
            raise ValueError(f"Size of expression_params: {self.n_expression_params}")

        # expand the params
        shape_params = params["shape_params"].expand(B, -1)
        expression_params = params["expression_params"].expand(B, -1)
        global_pose = params["global_pose"].expand(B, -1)
        neck_pose = params["neck_pose"].expand(B, -1)
        neck_pose = params["neck_pose"].expand(B, -1)
        jaw_pose = params["jaw_pose"].expand(B, -1)
        eye_pose = params["eye_pose"].expand(B, -1)
        transl = params["transl"].expand(B, -1)
        scale = params["scale"].expand(B, -1)

        # create the betas merged with shape and expression
        zero_shape = self.zero_shape.expand(B, -1)  # (B, 300 - S')
        shape = torch.cat([shape_params, zero_shape], dim=-1)  # (B, 300)
        zero_expression = self.zero_expression.expand(B, -1)  # (B, 100 - E')
        expression = torch.cat([expression_params, zero_expression], dim=-1)  # (B, 100)
        betas = torch.cat([shape, expression], dim=1)  # (B, 400)

        # create the pose merged with global, jaw, neck and left/right eye
        pose = torch.cat([global_pose, neck_pose, jaw_pose, eye_pose], dim=-1)  # (B,15)

        # apply the linear blend skinning model
        vertices, _ = lbs(
            betas,
            pose,
            self.v_template,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
        )  # (B, V, 3)

        # apply the translation and the scaling
        vertices += transl[:, None, :]  # (B, V, 3)
        vertices *= scale[:, None, :]  # (B, V, 3)

        # landmarks
        lm_vertices = vertices[:, self.lm_faces]  # (B, F, 3, D)
        lm_bary_coods = self.lm_bary_coords.expand(B, -1, -1).unsqueeze(-1)
        landmarks = (lm_bary_coods * lm_vertices).sum(-2)  # (B, 105, D)

        return {"vertices": vertices, "landmarks": landmarks}

    ####################################################################################
    # Model Utils
    ####################################################################################

    def set_params(self, **kwargs):
        """Initilize the params of the FLAME model.

        Args:
            **kwargs: dict(str, torch.Tensor): The key is the name of the nn.Embeddings
                table, which needs to be specified in order to override the initial
                values. The value can be a simple tensor of dim (D,) or the dimension of
                the nn.Embedding table (B, D).
        """
        for key, value in kwargs.items():
            if isinstance(value, list):
                value = torch.Tensor(value)
            value = value.to(self.device)
            self.params[key] = torch.nn.Parameter(value)

    def generate_default_params(self):
        shape_params = self.n_shape_params
        expression_params = self.n_expression_params
        params = {
            "global_pose": torch.zeros(3, device=self.device),
            "transl": torch.zeros(3, device=self.device),
            "jaw_pose": torch.zeros(3, device=self.device),
            "neck_pose": torch.zeros(3, device=self.device),
            "eye_pose": torch.zeros(6, device=self.device),
            "scale": torch.ones(1, device=self.device),
            "shape_params": torch.zeros(shape_params, device=self.device),
            "expression_params": torch.zeros(expression_params, device=self.device),
        }
        return {k: nn.Parameter(p.unsqueeze(0)) for k, p in params.items()}

    def mask_faces(self, faces: torch.Tensor, vertices_mask: torch.Tensor | None):
        """Calculates the triangular faces mask based on the masked vertices."""
        if vertices_mask is None:
            return faces
        vertices_mask = vertices_mask.expand(*faces.shape, -1)
        face_mask = (faces.unsqueeze(-1) == vertices_mask).any(dim=-1).all(dim=-1)
        return faces[face_mask]

    def extract_landmarks(self, landmarks: torch.Tensor):
        if landmarks.shape[1] != 105:
            return landmarks[:, self.lm_mediapipe_idx]
        return landmarks

    def render(self, renderer: Renderer, params: dict):
        m_out = self.forward(**params)
        r_out = renderer.render_full(
            vertices=m_out["vertices"],  # (B, V, 3)
            faces=self.faces,  # (F, 3)
        )
        return r_out
