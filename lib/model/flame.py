from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib import cm
from PIL import Image

from lib.model.lbs import lbs
from lib.model.loss import (
    calculate_point2plane,
    calculate_point2point,
    chamfer_distance,
    landmark_2d_distance,
    landmark_3d_distance,
)
from lib.rasterizer import Rasterizer
from lib.renderer.camera import Camera
from lib.renderer.renderer import Renderer
from lib.utils.loader import (
    load_flame,
    load_flame_masks,
    load_static_landmark_embedding,
)


class FLAME(L.LightningModule):
    def __init__(
        self,
        # model settings
        flame_dir: str = "/flame",
        num_shape_params: int = 100,
        num_expression_params: int = 50,
        # optimization settings
        lr: float = 1e-03,
        scheduler=None,
        optimize_frames: int = 1,  # the number of different frames per shape
        optimize_shapes: int = 1,  # the number of different shapes (allways 1)
        vertices_mask: str = "face",  # full, face
        # debug settings
        save_interval: int = 50,
        init_mode: str = "kinect",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["renderer"])

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
        self.scale = self.create_embeddings(torch.ones(optimize_frames, 3))

        # load the faces, mean vertices and pca bases
        self.faces = nn.Parameter(flame_model["f"], requires_grad=False)  # (9976, 3)
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

        # optimize modes
        self.optimize_modes: list[str] = ["default"]
        self.set_optimize_mode("default")

        # set optimization parameters
        self.optimization_parameters = [
            "shape_params",
            "expression_params",
            "global_pose",
            "neck_pose",
            # "jaw_pose",
            "eye_pose",
            "transl",
            "scale",
        ]

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
        default_frame_idx: int = 0,
        default_shape_idx: int = 0,
    ):
        """The forward pass the FLAME model.

        The vertices of the FLAME model are in world space. Hence the translation and
        the scaling are in world space.

        Args:
            shape_params (torch.tensor): (B, S') where S' is the number that was
                in the initialization and storedself.hparams["shape_params"])
            expression_params (torch.tensor): (B, E') where E' is the number that was
                in the initialization and storedself.hparams["expression_params"])
            global_pose (torch.tensor): number of global pose parameters (B, 3)
            neck_pose (torch.tensor): number of neck pose parameters (B, 3)
            jaw_pose (torch.tensor): number of jaw pose parameters (B, 3)
            eye_pose (torch.tensor): number of eye pose parameters (B, 6)
            transl(torch.tensor): number of translation parameters (B, 3)
            scale (torch.tensor): number of scale parameters (B, 3)

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
        if scale is None:
            scale = self.scale(frame_idx)

        # create the betas merged with shape and expression
        B = shape_params.shape[0]
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

        return vertices, landmarks  # (B, V, 3), (B, 105, 3)

    def model_step(self, frame_idx: torch.Tensor | int, shape_idx: torch.Tensor | int):
        if isinstance(frame_idx, int):
            frame_idx = torch.tensor([frame_idx], device=self.device)
        if isinstance(shape_idx, int):
            shape_idx = torch.tensor([shape_idx], device=self.device)
        vertices, landmarks = self.forward(
            shape_params=self.shape_params(shape_idx),  # (B, S')
            expression_params=self.expression_params(frame_idx),  # (B, E')
            global_pose=self.global_pose(frame_idx),
            neck_pose=self.neck_pose(frame_idx),
            jaw_pose=self.jaw_pose(frame_idx),
            eye_pose=self.eye_pose(frame_idx),
            transl=self.transl(frame_idx),
            scale=self.scale(frame_idx),
        )  # (B, V, 3)
        lm_3d_homo = self.renderer.camera.convert_to_homo_coords(landmarks)
        lm_2d_ndc_homo = self.renderer.camera.ndc_transform(lm_3d_homo)
        lm_2d_ndc = lm_2d_ndc_homo[..., :2]
        return {"vertices": vertices, "lm_3d_camera": landmarks, "lm_2d_ndc": lm_2d_ndc}

    def render_step(self, model: dict[str, torch.Tensor]):
        # render the current flame model
        faces = self.masked_faces(model["vertices"])  # (F', 3)
        render = self.renderer.render_full(
            vertices=model["vertices"],  # (B, V, 3)
            faces=faces,  # (F, 3)
        )  # (B, H', W')
        return render

    def on_train_start(self):
        # please here the initilizations that need some state from outside the model
        camera = getattr(self.trainer.datamodule, "camera", None)
        rasterizer = getattr(self.trainer.datamodule, "rasterizer", None)
        self.init_renderer(camera=camera, rasterizer=rasterizer)

    def training_step(self, batch, batch_idx):
        return self.optimization_step(batch, batch_idx)

    def optimization_step(self):
        raise NotImplementedError()

    def configure_optimizers(self):
        # 6 DoF initialization
        optimizer_params = {
            "params": filter(lambda p: p.requires_grad, self.parameters()),
            "lr": self.hparams["lr"],
        }
        optimizer = torch.optim.Adam(**optimizer_params)
        if self.hparams["scheduler"] is not None:
            scheduler = self.hparams["scheduler"](optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "train/loss"},
            }
        return {"optimizer": optimizer}

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

    def init_params_flame(self, sigma: float = 0.01, seed: int = 1):
        np.random.seed(seed)
        s = np.random.normal(0, sigma, 6)
        self.init_params(
            global_pose=[s[0], s[1], s[2]],
            transl=[0 + s[3], s[4], s[5] - 0.5],
        )

    def init_params_dphm(self, sigma: float = 0.01, seed: int = 1):
        np.random.seed(seed)
        s = np.random.normal(0, sigma, 6)
        self.init_params(
            global_pose=[s[0], s[1], s[2]],
            transl=[s[3], s[4], s[5] - 0.5],
        )

    def set_optimize_mode(self, mode: str):
        assert mode in self.optimize_modes
        self.optimize_mode = mode

    def create_embeddings(self, tensor: torch.Tensor):
        """Creates an embedding table for multi-view multi shape optimization."""
        num_embeddings, embedding_dim = tensor.shape
        return nn.Embedding(num_embeddings, embedding_dim, _weight=tensor)

    def masked_faces(self, vertices: torch.Tensor):
        """Calculates the triangular faces mask based on the masked vertices.

        Args:
            vertices (torch.Tensor): The vertices in camera coordinate system (B, V, 3)

        Returns:
            (torch.Tensor): A boolean mask of the faces that should be used for the
                computation, the final dimension of the mask is (F, 3).
        """
        if self.vertices_mask is None:
            return self.faces
        vertices_mask = self.vertices_mask.expand(*self.faces.shape, -1).to(self.device)
        face_mask = (self.faces.unsqueeze(-1) == vertices_mask).any(dim=-1).all(dim=-1)
        return self.faces[face_mask]

    def inlier_mask(
        self,
        batch: dict[str, torch.Tensor],
        render: dict[str, torch.Tensor],
        n_threshold: float = 0.9,
        d_threshold: float = 0.1,
    ):
        # per pixel distance in 3d, just the length
        dist = torch.norm(render["point"] - batch["point"], dim=-1)  # (B, W, H)
        # calculate the forground mask
        f_mask = batch["mask"] & render["mask"]  # (B, W, H)
        # the depth mask based on some epsilon of distance 10cm
        d_mask = dist < d_threshold  # (B, W, H)
        # dot product, e.g. coresponds to an angle
        normal_dot = (batch["normal"] * render["normal"]).sum(-1)
        n_mask = normal_dot > n_threshold  # (B, W, H)
        # final loss mask of silhouette, depth and normal threshold
        mask = d_mask & f_mask & n_mask
        assert mask.sum()  # we have some overlap
        return {"mask": mask, "f_mask": f_mask, "n_mask": n_mask, "d_mask": d_mask}

    def extract_landmarks(self, landmarks: torch.Tensor):
        if landmarks.shape[1] != 105:
            return landmarks[:, self.lm_mediapipe_idx]
        return landmarks

    ####################################################################################
    # Logging and Debuging Utils
    ####################################################################################

    def iter_debug_idx(self, batch: dict[str, torch.Tensor]):
        B = batch["frame_idx"].shape[0]
        for idx in range(B):
            yield B, idx, batch["frame_idx"][idx].item()

    def debug_loss(self, batch: dict, render: dict, mask: dict, max_loss=1e-02):
        """The max is an error of 10cm"""
        # color entcoding for the loss map
        norm = plt.Normalize(0.0, vmax=max_loss)
        cmap = plt.get_cmap("jet")
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)

        for _, b_idx, f_idx in self.iter_debug_idx(batch):
            # point to point error map
            file_name = f"error_point_to_point/{f_idx:05}/{self.current_epoch:05}.png"
            loss = torch.sqrt(
                calculate_point2point(batch["point"], render["point"])
            )  # (B, W, H)
            error_map = loss[b_idx].detach().cpu().numpy()  # dist in m
            error_map = torch.from_numpy(sm.to_rgba(error_map)[:, :, :3])
            error_map[~render["mask"][b_idx], :] = 1.0
            error_map = (error_map * 255).to(torch.uint8)
            self.save_image(file_name, error_map)

            # point to plane error map
            file_name = f"error_point_to_plane/{f_idx:05}/{self.current_epoch:05}.png"
            loss = torch.sqrt(
                calculate_point2plane(
                    p=render["point"],
                    q=batch["point"],
                    n=render["normal"],
                )
            )  # (B, W, H)
            error_map = loss[b_idx].detach().cpu().numpy()  # dist in m
            error_map = torch.from_numpy(sm.to_rgba(error_map)[:, :, :3])
            error_map[~render["mask"][b_idx], :] = 1.0
            error_map = (error_map * 255).to(torch.uint8)
            self.save_image(file_name, error_map)

            # error mask
            file_name = f"error_mask/{f_idx:05}/{self.current_epoch:05}.png"
            self.save_image(file_name, mask["mask"][b_idx])

    def debug_render(self, batch: dict, render: dict, model: dict):
        for _, b_idx, f_idx in self.iter_debug_idx(batch):
            file_name = f"render_mask/{f_idx:05}/{self.current_epoch:05}.png"
            self.save_image(file_name, render["mask"][b_idx])

            file_name = f"render_depth/{f_idx:05}/{self.current_epoch:05}.png"
            self.save_image(file_name, render["depth_image"][b_idx])

            file_name = f"render_normal/{f_idx:05}/{self.current_epoch:05}.png"
            self.save_image(file_name, render["normal_image"][b_idx])

            file_name = f"render_color/{f_idx:05}/{self.current_epoch:05}.png"
            self.save_image(file_name, render["color"][b_idx])

            file_name = f"render_merged_depth/{f_idx:05}/{self.current_epoch:05}.png"
            render_merged_depth = batch["color"].clone()
            color_mask = (
                (render["point"][:, :, :, 2] < batch["point"][:, :, :, 2])
                | (render["mask"] & ~batch["mask"])
            ) & render["mask"]
            render_merged_depth[color_mask] = render["color"][color_mask]
            self.save_image(file_name, render_merged_depth[b_idx])

            file_name = f"render_merged/{f_idx:05}/{self.current_epoch:05}.png"
            render_merged = batch["color"].clone()
            render_merged[render["mask"]] = render["color"][render["mask"]]
            self.save_image(file_name, render_merged[b_idx])

            file_name = f"render_landmark/{f_idx:05}/{self.current_epoch:05}.png"
            lm_2d_screen = self.renderer.camera.xy_ndc_to_screen(model["lm_2d_ndc"])
            x_idx = lm_2d_screen[b_idx, :, 0].to(torch.int32)
            y_idx = lm_2d_screen[b_idx, :, 1].to(torch.int32)
            color_image = render["color"][b_idx]
            color_image[y_idx, x_idx, :] = 255
            self.save_image(file_name, color_image)

    def debug_3d_points(self, batch: dict, render: dict, model: dict):
        for B, b_idx, f_idx in self.iter_debug_idx(batch):
            # save the gt points
            file_name = f"point_batch/{f_idx:05}/{self.current_epoch:05}.npy"
            points = batch["point"][b_idx][batch["mask"][b_idx]].reshape(-1, 3)
            self.save_points(file_name, points)
            # save the flame points
            file_name = f"point_render/{f_idx:05}/{self.current_epoch:05}.npy"
            render_points = render["point"][b_idx][render["mask"][b_idx]].reshape(-1, 3)
            self.save_points(file_name, render_points[b_idx])
            # save the lm_3d gt points
            file_name = f"point_batch_landmark/{f_idx:05}/{self.current_epoch:05}.npy"
            self.save_points(file_name, batch["lm_3d_camera"][b_idx])
            # save the lm_3d flame points
            file_name = f"point_render_landmark/{f_idx:05}/{self.current_epoch:05}.npy"
            self.save_points(file_name, model["lm_3d_camera"][b_idx])

    def debug_input_batch(self, batch: dict, model: dict, render: dict):
        for _, b_idx, f_idx in self.iter_debug_idx(batch):
            file_name = f"batch_mask/{f_idx:05}/{self.current_epoch:05}.png"
            self.save_image(file_name, batch["mask"][b_idx])

            file_name = f"batch_depth/{f_idx:05}/{self.current_epoch:05}.png"
            depth = Renderer.point_to_depth(batch["point"])
            depth_image = Renderer.depth_to_depth_image(depth)
            self.save_image(file_name, depth_image[b_idx])

            file_name = f"batch_normal/{f_idx:05}/{self.current_epoch:05}.png"
            normal_image = Renderer.normal_to_normal_image(
                batch["normal"], batch["mask"]
            )
            self.save_image(file_name, normal_image[b_idx])

            file_name = f"batch_color/{f_idx:05}/{self.current_epoch:05}.png"
            self.save_image(file_name, batch["color"][b_idx])

            file_name = f"batch_landmark/{f_idx:05}/{self.current_epoch:05}.png"
            lm_2d_screen = self.renderer.camera.xy_ndc_to_screen(batch["lm_2d_ndc"])
            x_idx = lm_2d_screen[b_idx, :, 0].to(torch.int32)
            y_idx = lm_2d_screen[b_idx, :, 1].to(torch.int32)
            color_image = batch["color"][b_idx]
            color_image[y_idx, x_idx, :] = 255
            self.save_image(file_name, color_image)

    def debug_metrics(self, batch: dict, model: dict, render: dict, mask: dict):
        chamfers = []
        for _, b_idx, _ in self.iter_debug_idx(batch):
            # debug chamfer loss
            chamfer = chamfer_distance(
                render["point"][b_idx][render["mask"][b_idx]].reshape(1, -1, 3),
                batch["point"][b_idx][batch["mask"][b_idx]].reshape(1, -1, 3),
            )  # (B,)
            chamfers.append(chamfer)
        self.log("train/chamfer", torch.stack(chamfers).mean(), prog_bar=True)
        # debug lm_3d loss, the dataset contains all >400 mediapipe landmarks
        lm_3d_camera = self.extract_landmarks(batch["lm_3d_camera"])
        lm_3d_dist = landmark_3d_distance(model["lm_3d_camera"], lm_3d_camera)  # (B,)
        self.log("train/lm_3d_loss", lm_3d_dist.mean(), prog_bar=True)
        # debug lm_3d loss, the dataset contains all >400 mediapipe landmarks
        lm_2d_ndc = self.extract_landmarks(batch["lm_2d_ndc"])
        lm_2d_dist = landmark_2d_distance(model["lm_2d_ndc"], lm_2d_ndc)  # (B,)
        self.log("train/lm_2d_loss", lm_2d_dist.mean(), prog_bar=True)
        # debug the overlap of the mask
        for key, value in mask.items():
            self.log(f"debug/{key}", float(value.sum()))

    def debug_params(self, batch: dict):
        for p_name in self.optimization_parameters:
            param = getattr(self, p_name, None)
            if p_name in batch and param is not None:
                if p_name in ["shape_params"]:
                    weight = param(batch["shape_idx"])
                else:
                    weight = param(batch["frame_idx"])
                param_dist = torch.norm(batch[p_name] - weight, dim=-1).mean()
                self.log(f"debug/param_{p_name}_l2", param_dist, prog_bar=False)

    def debug_gradients(self, optimizer):
        for p_name in self.optimization_parameters:
            param = getattr(self, p_name, None)
            if param is None or not param.weight.requires_grad:
                continue
            for i in range(param.weight.shape[-1]):
                p = param.weight[:, i].mean()
                self.log_dict({f"debug/{p_name}_{i}": p})
            for i in range(param.weight.grad.shape[-1]):
                p = param.weight.grad[:, i].mean()
                self.log_dict({f"debug/{p_name}_{i}_grad": p})
            self.log_dict({f"debug/{p_name}_mean": param.weight.mean()})
            self.log_dict({f"debug/{p_name}_absmax": param.weight.abs().max()})
            self.log_dict({f"debug/{p_name}_mean_grad": param.weight.grad.mean()})
            self.log_dict(
                {f"debug/{p_name}_absmax_grad": param.weight.grad.abs().max()}
            )

    def save_image(self, file_name: str, image: torch.Tensor):
        path = Path(self.logger.save_dir) / file_name  # type: ignore
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image.detach().cpu().numpy()).save(path)

    def save_points(self, file_name: str, points: torch.Tensor):
        path = Path(self.logger.save_dir) / file_name  # type: ignore
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            np.save(f, points.detach().cpu().numpy())


########################################################################################
# FLAMEPoint2Point
########################################################################################


class FLAMEPoint2Point(FLAME):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.hparams["init_mode"] == "kinect":
            self.init_params_dphm(0.01, seed=2)
        else:
            self.init_params_flame(0.03, seed=2)

    def optimization_step(self, batch, batch_idx):
        # forward pass with the current frame and shape
        model = self.model_step(
            frame_idx=batch["frame_idx"],  # (B, )
            shape_idx=batch["shape_idx"],  # (B, )
        )
        # render the current flame model
        render = self.render_step(model)
        # rejection based on outliers
        mask = self.inlier_mask(batch=batch, render=render)
        # distance in camera space per pixel & point to point loss
        p2p = calculate_point2point(batch["point"], render["point"])  # (B, W, H)
        # point to point loss
        loss = p2p[mask["mask"]].mean()
        self.log("train/point2point", loss, prog_bar=True)
        # log metrics
        # self.debug_metrics(batch=batch, model=model, render=render, mask=mask)
        if (self.current_epoch % self.hparams["save_interval"]) == 0:
            # self.debug_3d_points(batch=batch, model=model, render=render)
            self.debug_render(batch=batch, model=model, render=render)
            self.debug_input_batch(batch=batch, model=model, render=render)
            self.debug_loss(batch=batch, render=render, mask=mask)
            if self.hparams["init_mode"] == "flame":
                self.debug_params(batch=batch)

        return loss

    def on_before_optimizer_step(self, optimizer):
        self.debug_gradients(optimizer)


########################################################################################
# FLAMEPoint2Plane
########################################################################################


class FLAMEPoint2Plane(FLAME):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.hparams["init_mode"] == "kinect":
            self.init_params_dphm(0.01, seed=2)
        else:
            self.init_params_flame(0.03, seed=2)

    def optimization_step(self, batch, batch_idx):
        # forward pass to get the data
        model = self.model_step(
            frame_idx=batch["frame_idx"],  # (B, )
            shape_idx=batch["shape_idx"],  # (B, )
        )
        render = self.render_step(model)
        mask = self.inlier_mask(batch=batch, render=render)

        point2plane = calculate_point2plane(
            q=render["point"],
            p=batch["point"].detach(),
            n=render["normal"].detach(),
        )  # (B, W, H)
        point2plane_loss = point2plane[mask["mask"]].mean()
        self.log("train/point2plane", point2plane_loss, prog_bar=True)

        # point2point = calculate_point2point(
        #     q=render["point"],
        #     p=batch["point"].detach(),
        # )  # (B, W, H)
        # point2point_loss = 0.1 * point2point[mask["mask"]].mean()
        # self.log("train/point2point", point2point_loss, prog_bar=True)

        # lm_2d = landmark_2d_distance(
        #     model["lm_2d_ndc"],
        #     self.extract_landmarks(batch["lm_2d_ndc"]).detach(),
        # )  # (B, L)
        # lm_2d_loss = lm_2d.mean()

        # final loss
        loss = point2plane_loss
        self.log("train/loss", loss, prog_bar=True)

        # debug and logging
        self.debug_metrics(batch=batch, model=model, render=render, mask=mask)
        if (self.current_epoch % self.hparams["save_interval"]) == 0:
            # self.debug_3d_points(batch=batch, model=model, render=render)
            self.debug_render(batch=batch, model=model, render=render)
            self.debug_input_batch(batch=batch, model=model, render=render)
            self.debug_loss(batch=batch, render=render, mask=mask)
            if self.hparams["init_mode"] == "flame":
                self.debug_params(batch=batch)

        return loss

    def on_before_optimizer_step(self, optimizer):
        self.debug_gradients(optimizer)
