import numpy as np
import torch
import torch.nn as nn

from lib.data.loader import load_flame, load_flame_masks, load_static_landmark_embedding
from lib.model.lbs import lbs
from lib.rasterizer import Rasterizer
from lib.renderer.camera import Camera
from lib.renderer.renderer import Renderer


class FLAME(nn.Module):
    def __init__(
        self,
        flame_dir: str = "/flame",
        num_shape_params: int = 100,
        num_expression_params: int = 50,
        optimize_frames: int = 1,
        optimize_shapes: int = 1,
        vertices_mask: str = "face",  # full, face
        n_threshold: float = 0.9,
        d_threshold: float = 0.1,
        init_config: dict = {},
        **kwargs,
    ):
        super().__init__()

        # load the face model
        flame_model = load_flame(flame_dir=flame_dir, return_tensors="pt")

        # shape parameters, default just 1 shape per optimization
        zero_shape = torch.zeros(300 - num_shape_params)
        self.zero_shape = nn.Parameter(zero_shape)
        shape_params = torch.zeros(optimize_shapes, num_shape_params)
        self.shape_params = self.create_embeddings(shape_params)

        # expression parameters
        zero_expression = torch.zeros(100 - num_expression_params)
        self.zero_expression = nn.Parameter(zero_expression)
        expression_params = torch.zeros(optimize_frames, num_expression_params)
        self.expression_params = self.create_embeddings(expression_params)

        # pose parameters
        self.global_pose = self.create_embeddings(torch.zeros(optimize_frames, 3))
        self.neck_pose = self.create_embeddings(torch.zeros(optimize_frames, 3))
        self.jaw_pose = self.create_embeddings(torch.zeros(optimize_frames, 3))
        self.eye_pose = self.create_embeddings(torch.zeros(optimize_frames, 6))

        # translation and scale pose
        self.transl = self.create_embeddings(torch.zeros(optimize_frames, 3))

        # load the faces
        self.faces = nn.Parameter(flame_model["f"], requires_grad=False)  # (9976, 3)
        face_mask = load_flame_masks(flame_dir, return_tensors="pt")[vertices_mask]
        masked_faces = self.mask_faces(flame_model["f"], face_mask)
        self.masked_faces = torch.nn.Parameter(masked_faces, requires_grad=False)

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

        # corresponding threshold
        self.n_threshold = n_threshold
        self.d_threshold = d_threshold

        # set optimization parameters
        self.shape_p_names = ["shape_params"]
        self.frame_p_names = [
            "global_pose",
            "transl",
            "neck_pose",
            "eye_pose",
            "expression_params",
        ]
        self.full_p_names = self.shape_p_names + self.frame_p_names

        # init the params of the model
        self.init_config = init_config
        self.init_params_with_config(init_config=init_config)

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
        default_frame_idx: int = 0,
        default_shape_idx: int = 0,
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

        Return:
            (torch.Tensor): The mesh vertices of dim (B, V, 3)
        """
        # select default params if None was selected
        shape_idx = torch.tensor([default_shape_idx], device=self.device)
        frame_idx = torch.tensor([default_frame_idx], device=self.device)
        if shape_params is None:
            shape_params = self.shape_params(shape_idx)  # (B, S')
        if expression_params is None:
            expression_params = self.expression_params(frame_idx)  # (B, E')
        if global_pose is None:
            global_pose = self.global_pose(frame_idx)
        if neck_pose is None:
            neck_pose = self.neck_pose(frame_idx)
        if jaw_pose is None:
            jaw_pose = self.jaw_pose(frame_idx)
        if eye_pose is None:
            eye_pose = self.eye_pose(frame_idx)
        if transl is None:
            transl = self.transl(frame_idx)

        # create the betas merged with shape and expression
        B = global_pose.shape[0]
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

        # landmarks
        lm_vertices = vertices[:, self.lm_faces]  # (B, F, 3, D)
        lm_bary_coods = self.lm_bary_coords.expand(B, -1, -1).unsqueeze(-1)
        landmarks = (lm_bary_coods * lm_vertices).sum(-2)  # (B, 105, D)

        # extract the 2D landmark locations
        lm_3d_homo = self.renderer.camera.convert_to_homo_coords(landmarks)
        lm_2d_ndc_homo = self.renderer.camera.ndc_transform(lm_3d_homo)
        lm_2d_ndc = lm_2d_ndc_homo[..., :2]

        return {"vertices": vertices, "lm_3d_camera": landmarks, "lm_2d_ndc": lm_2d_ndc}

    def correspondence_step(self, batch: dict):
        self.logger.time_tracker.start("model_forward")
        model = self.forward(**self.flame_input_dict(batch))
        self.logger.time_tracker.start("renderer", stop=True)
        render = self.renderer.render_full(
            vertices=model["vertices"],  # (B, V, 3)
            faces=self.masked_faces,  # (F, 3)
        )  # (B, H', W')
        self.logger.time_tracker.start("mask", stop=True)
        # per pixel distance in 3d, just the length
        dist = torch.norm(render["point"] - batch["point"], dim=-1)  # (B, W, H)
        # calculate the forground mask
        f_mask = batch["mask"] & render["r_mask"]  # (B, W, H)
        # the depth mask based on some epsilon of distance 10cm
        d_mask = dist < self.d_threshold  # (B, W, H)
        # dot product, e.g. coresponds to an angle
        normal_dot = (batch["normal"] * render["normal"]).sum(-1)
        n_mask = normal_dot > self.n_threshold  # (B, W, H)
        # final loss mask of silhouette, depth and normal threshold
        final_mask = d_mask & f_mask & n_mask
        assert final_mask.sum()  # we have some overlap
        self.logger.time_tracker.stop()
        mask = {
            "mask": final_mask,
            "f_mask": f_mask,
            "n_mask": n_mask,
            "d_mask": d_mask,
        }
        return {**model, **render, **mask}

    def model_step(
        self,
        batch: dict,
        correspondences: dict,
        params: list[torch.Tensor],
        p_names: list[str],
    ):
        B = batch["frame_idx"].shape[0]
        flame_input = self.flame_input_dict(batch)
        flame_params = {}
        for p_name, param in flame_input.items():
            flame_params[p_name] = param.expand(B, -1)
        for p_name, param in zip(p_names, params):
            flame_params[p_name] = param.expand(B, -1)

        mask = correspondences["mask"]
        out = self.forward(**flame_params)
        point = self.renderer.mask_interpolate(
            vertices_idx=correspondences["vertices_idx"],
            bary_coords=correspondences["bary_coords"],
            attributes=out["vertices"],
            mask=mask,
        )  # (C, 3)

        # set the params for regularization
        shape_params = None
        expression_params = None
        for p_name, param in zip(p_names, params):
            if p_name == "shape_params":
                shape_params = param
            if p_name == "expression_params":
                expression_params = param

        return {
            "mask": mask,
            "point": point,
            "normal": correspondences["normal"][mask],  # (C, 3)
            "point_gt": batch["point"][mask],  # (C, 3)
            "normal_gt": batch["normal"][mask],  # (C, 3)
            "shape_params": shape_params,  # (B, 100)
            "expression_params": expression_params,  # (B, 100)
            "lm_3d_camera": out["lm_3d_camera"],
            "lm_2d_ndc": out["lm_2d_ndc"],
            "lm_3d_camera_gt": self.extract_landmarks(batch["lm_3d_camera"]),
            "lm_2d_ndc_gt": self.extract_landmarks(batch["lm_2d_ndc"]),
        }

    def model_stepv3(
        self,
        batch: dict,
        correspondences: dict,
        params: torch.Tensor,
        p_names: list[str],
    ):
        B = batch["frame_idx"].shape[0]
        flame_input = self.flame_input_dict(batch)
        flame_params = {}
        for p_name, param in flame_input.items():
            flame_params[p_name] = param.expand(B, -1)

        idx = 0
        for p_name in p_names:
            offset = flame_input[p_name].numel()
            flame_params[p_name] = params[idx : idx + offset].expand(B, -1)
            idx += offset

        mask = correspondences["mask"]
        out = self.forward(**flame_params)
        point = self.renderer.mask_interpolate(
            vertices_idx=correspondences["vertices_idx"],
            bary_coords=correspondences["bary_coords"],
            attributes=out["vertices"],
            mask=mask,
        )  # (C, 3)

        # set the params for regularization
        shape_params = None
        expression_params = None
        for p_name, param in zip(p_names, params):
            if p_name == "shape_params":
                shape_params = param
            if p_name == "expression_params":
                expression_params = param

        return {
            "mask": mask,
            "point": point,
            "normal": correspondences["normal"][mask],  # (C, 3)
            "point_gt": batch["point"][mask],  # (C, 3)
            "normal_gt": batch["normal"][mask],  # (C, 3)
            "shape_params": shape_params,  # (B, 100)
            "expression_params": expression_params,  # (B, 100)
            "lm_3d_camera": out["lm_3d_camera"],
            "lm_2d_ndc": out["lm_2d_ndc"],
            "lm_3d_camera_gt": self.extract_landmarks(batch["lm_3d_camera"]),
            "lm_2d_ndc_gt": self.extract_landmarks(batch["lm_2d_ndc"]),
        }

    ####################################################################################
    # Model Utils
    ####################################################################################

    def init_renderer(
        self,
        camera: Camera,
        rasterizer: Rasterizer,
        **kwargs,
    ):
        # copy camera and rasterizer from the datamodule
        renderer = getattr(self, "renderer", None)
        if renderer is None and camera is not None and rasterizer is not None:
            self.renderer = Renderer(camera=camera, rasterizer=rasterizer, **kwargs)

    def init_params(self, **kwargs):
        """Initilize the params of the FLAME model.

        Args:
            **kwargs: dict(str, torch.Tensor): The key is the name of the nn.Embeddings
                table, which needs to be specified in order to override the initial
                values. The value can be a simple tensor of dim (D,) or the dimension of
                the nn.Embedding table (B, D).
        """
        for key, value in kwargs.items():
            param = self.__getattr__(key)
            if isinstance(value, list):
                value = torch.Tensor(value)
            value = value.expand(param.weight.shape).to(self.device)
            param.weight = torch.nn.Parameter(value, requires_grad=True)

    def init_params_with_config(self, init_config: dict):
        seed = init_config.get("seed")
        sigma = init_config.get("sigma")
        assert sigma is not None
        params = {}
        for p_name in self.full_p_names:
            if param := init_config.get(p_name):
                np.random.seed(seed)
                delta = np.random.normal(0, sigma, len(param))
                params[p_name] = [p + d for p, d in zip(param, delta)]
        self.init_params(**params)

    def init_frame(self, target_idx: int, source_idx: int):
        assert source_idx >= 0
        assert target_idx >= 0
        for p_name in self.frame_p_names:
            module = getattr(self, p_name)
            module.weight[target_idx] = module.weight[source_idx].detach()

    def init_logger(self, logger):
        self.logger = logger
        self.renderer.init_logger(logger)

    def reset_frame(self, frame_idx: int):
        for p_name in self.frame_p_names:
            module = getattr(self, p_name)
            module.weight[frame_idx] = torch.zeros_like(module.weight[frame_idx])

    def create_embeddings(self, tensor: torch.Tensor):
        """Creates an embedding table for multi-view multi shape optimization."""
        num_embeddings, embedding_dim = tensor.shape
        return nn.Embedding(num_embeddings, embedding_dim, _weight=tensor)

    def mask_faces(self, faces: torch.Tensor, vertices_mask: torch.Tensor):
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

    @property
    def device(self):
        return self.shape_params.weight.device

    def flame_input_dict(self, batch: dict):
        return dict(
            shape_params=self.shape_params(batch["shape_idx"]),  # (B, S')
            expression_params=self.expression_params(batch["frame_idx"]),  # (B, E')
            global_pose=self.global_pose(batch["frame_idx"]),
            neck_pose=self.neck_pose(batch["frame_idx"]),
            jaw_pose=self.jaw_pose(batch["frame_idx"]),
            eye_pose=self.eye_pose(batch["frame_idx"]),
            transl=self.transl(batch["frame_idx"]),
        )
