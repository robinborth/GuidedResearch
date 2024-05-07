from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from lib.model.lbs import lbs
from lib.model.loss import landmark_3d_loss, point_to_point_loss
from lib.renderer.renderer import Renderer
from lib.utils.loader import (
    load_flame,
    load_flame_masks,
    load_intrinsics,
    load_static_landmark_embedding,
)
from lib.utils.logger import create_logger

log = create_logger("flame")


class FLAME(L.LightningModule):
    def __init__(
        self,
        # model settings
        flame_dir: str = "/flame",
        data_dir: str = "/data",
        num_shape_params: int = 100,
        num_expression_params: int = 50,
        # optimization settings
        lr: float = 1e-03,
        scheduler=None,
        lm_weight: float = 1.0,
        p2p_weight: float = 0.1,
        optimize_frames: int = 1,  # the number of different frames per shape
        optimize_shapes: int = 1,  # the number of different shapes (allways 1)
        vertices_mask: bool = False,
        # debug settings
        renderer: Renderer | None = None,
        save_interval: int = 50,
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
        self.vertices_mask = None
        if vertices_mask:
            face_mask = load_flame_masks(flame_dir, return_tensors="pt")["face"]
            self.vertices_mask = torch.nn.Parameter(face_mask, requires_grad=False)

        # flame landmarks to guide sparse landmark loss
        lms = load_static_landmark_embedding(flame_dir, return_tensors="pt")
        lm_faces = self.faces[lms["lm_face_idx"]]  # (105, )
        self.lm_faces = torch.nn.Parameter(lm_faces, requires_grad=False)
        lm_bary_coords = lms["lm_bary_coords"]  # (105, 3)
        self.lm_bary_coords = torch.nn.Parameter(lm_bary_coords, requires_grad=False)
        lm_idx = lms["lm_mediapipe_idx"]  # (105,)
        self.lm_mediapipe_idx = torch.nn.Parameter(lm_idx, requires_grad=False)

        # load the default intriniscs or initialize it with something good
        self.K = load_intrinsics(data_dir=data_dir, return_tensor="pt")

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

    def training_step(self, batch, batch_idx):
        return self.optimization_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer_params = self.optimization_params()
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

    def create_embeddings(self, tensor: torch.Tensor):
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
        vertices_mask = self.vertices_mask.expand(*self.faces, -1).to(self.device)
        face_mask = (self.faces.unsqueeze(-1) == vertices_mask).any(dim=-1).all(dim=-1)
        return self.faces[face_mask]

    def optimization_params(self):
        raise NotImplementedError()

    def optimization_step(self):
        raise NotImplementedError()

    def on_before_optimizer_step(self, optimizer):
        self.log_dict({"debug/transl_x": self.transl.weight[0, 0]})
        transl_grad = optimizer.param_groups[0]["params"][0][0]
        self.log_dict({"debug/transl_x_grad": transl_grad[0]})

        self.log_dict({"debug/transl_y": self.transl.weight[0, 1]})
        transl_grad = optimizer.param_groups[0]["params"][0][0]
        self.log_dict({"debug/transl_y_grad": transl_grad[1]})

        self.log_dict({"debug/transl_z": self.transl.weight[0, 2]})
        transl_grad = optimizer.param_groups[0]["params"][0][0]
        self.log_dict({"debug/transl_z_grad": transl_grad[2]})

    ####################################################################################
    # Dataset Utils Coarse To Fine
    ####################################################################################

    @property
    def image_scale(self):
        if "image_scale" in self.hparams:
            return self.hparams["image_scale"]
        return self.trainer.datamodule.image_scale

    @property
    def image_height(self):
        if "image_height" in self.hparams:
            return self.hparams["image_height"]
        return self.trainer.datamodule.image_height

    @property
    def image_width(self):
        if "image_width" in self.hparams:
            return self.hparams["image_width"]
        return self.trainer.datamodule.image_width

    def renderer(self, **kwargs):
        return Renderer(
            K=self.K,
            image_scale=self.image_scale,
            image_height=self.image_height,
            image_width=self.image_width,
            device=self.device,
            **kwargs,
        )

    ####################################################################################
    # Logging and Debuging Utils
    ####################################################################################
    def error_images(self, loss: torch.Tensor, mask: torch.Tensor):
        """The error image per pixel.

        Args:
            loss (torch.Tensor): The loss image per pixel of dim (B, H, W)
            mask (torch.Tensor): The mask to set the error to zero (B, H, W)

        Returns:
            (torch.Tensor): The error image as pixel values ranging from
                0 ... 255, where 255 means a high error of dim (B, H, W, 3)
        """
        error_image = torch.zeros((*loss.shape, 3), device=loss.device)
        error_image[:, :, :, 0] = loss * 4
        error_image = (error_image.clip(0, 1) * 255).to(torch.uint8)
        error_image[~mask] = 0
        return error_image

    def points_direction_image(
        self,
        s_points: torch.Tensor,
        t_points: torch.Tensor,
        mask: torch.Tensor,
    ):
        point_dir = t_points - s_points
        point_dir = point_dir / torch.norm(point_dir, dim=-1).unsqueeze(-1)
        point_dir = (((point_dir + 1) / 2) * 255).to(torch.uint8)
        point_dir[~mask, :] = 0
        return point_dir

    def save_image(self, file_name: str, image: torch.Tensor):
        path = Path(self.logger.save_dir) / file_name  # type: ignore
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image.detach().cpu().numpy()).save(path)

    def save_points(self, file_name: str, points: torch.Tensor):
        path = Path(self.logger.save_dir) / file_name  # type: ignore
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            np.save(f, points.detach().cpu().numpy())


class FLAMEPoints(FLAME):
    # NOTE: The global and trans have a custom initialization e.g. becaue this is
    # how the camera is positioned, e.g. we need to an overlapping in oder to compute
    # the loss
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        x_eps = 0.04
        y_eps = -0.02
        z_eps = 0.03

        alpha_eps = 0.03
        beta_eps = -0.05
        gamma_eps = 0.06

        self.init_params(
            global_pose=[torch.pi + alpha_eps, 0.0 + beta_eps, 0.0 + gamma_eps],
            transl=[0.0 + x_eps, 0.27 + y_eps, 0.5 + z_eps],
        )

    def model_step(self, batch, batch_idx):
        ...
    
    def debug_step(self, batch, batch_idx):
        ...

    def optimization_step(self, batch, batch_idx):
        frame_idx = batch["frame_idx"]  # (B, )
        shape_idx = batch["shape_idx"]  # (B, )
        B = batch["frame_idx"].shape[0]  # batch_size
        b_idx = 0
        f_idx = batch["frame_idx"][b_idx].item()

        point = batch["point"]  # (B, H', W' 3)
        mask = batch["mask"]  # (B, H', W') forground
        normal = batch["normal"]  # (B, H', W' 3)
        color = batch["color"]  # (B, H', W', 3)
        # lm3ds = batch["lm3ds"][:, self.lm_mediapipe_idx]  # (B, 105, 3)  # TODO this is because of FLAME DATASET
        lm3ds = batch["lm3ds"]  # (B, 105, 3)

        vertices, lm3ds_hat = self.forward(
            shape_params=self.shape_params(shape_idx),  # (B, S')
            expression_params=self.expression_params(frame_idx),  # (B, E')
            global_pose=self.global_pose(frame_idx),
            neck_pose=self.neck_pose(frame_idx),
            jaw_pose=self.jaw_pose(frame_idx),
            eye_pose=self.eye_pose(frame_idx),
            transl=self.transl(frame_idx),
            scale=self.scale(frame_idx),
        )  # (B, V, 3)

        renderer = renderer = self.renderer(
            diffuse=[0.0, 0.0, 0.6],
            specular=[0.0, 0.0, 0.5],
        )

        faces = self.masked_faces(vertices)  # (F', 3)
        render = renderer.render_full(
            vertices=vertices,  # (B, V, 3)
            faces=faces,  # (F, 3)
        )  # (B, H', W')

        # per pixel distance in 3d, just the length
        dist = torch.norm(render["point"] - point, dim=-1)  # (B, W, H)

        # calculate the forground mask
        f_mask = mask & render["mask"]  # (B, W, H)
        assert f_mask.sum()  # we have some overlap
        self.log("debug/n_f_mask", float(f_mask.sum()))

        # the depth mask based on some epsilon of distance
        d_threshold = 0.1  # 10cm
        d_mask = dist < d_threshold  # (B, W, H)
        self.log("debug/n_d_mask", float(d_mask.sum()))

        n_threshold = 0.8  # this is the dot product, e.g. coresponds to an angle
        normal_dot = (normal * render["normal"]).sum(-1)
        n_mask = normal_dot > n_threshold  # (B, W, H)
        self.log("debug/n_n_mask", float(n_mask.sum()))

        # final loss mask of silhouette, depth and normal threshold
        l_mask = d_mask & f_mask & n_mask
        self.log("debug/n_mask", float(mask.sum()))

        # compute the loss based on the distance, mean of squared distances
        p2p_loss = torch.pow(dist[l_mask], 2).mean()
        self.log("train/p2p_loss", p2p_loss)        

        loss = p2p_loss
        self.log("train/loss", loss, prog_bar=True)

        # debug chamfer loss
        chamfer = point_to_point_loss(
            vertices=vertices,
            points=point[mask].reshape(B, -1, 3),
        )  # (B,)
        self.log("train/chamfer", chamfer.mean(), prog_bar=True)

        # debug landmark loss
        lm_loss = landmark_3d_loss(lm3ds_hat, lm3ds)  # (B,)
        self.log("train/lm_loss", lm_loss.mean(), prog_bar=True)

        # visualize the image that is optimized and save to disk and the 3d points
        if (self.current_epoch + 0) % self.hparams["save_interval"] == 0:
            file_name = f"error/{f_idx:03}_{self.current_epoch:05}.png"
            error_image = self.error_images(dist, f_mask)  # (B, H, W, 3)
            self.save_image(file_name, error_image[b_idx])

            file_name = f"points_direction/{f_idx:03}_{self.current_epoch:05}.png"
            points_direction_image = self.points_direction_image(
                point, render["point"], render["mask"]
            )
            self.save_image(file_name, points_direction_image[b_idx])

            file_name = f"render_mask/{f_idx:03}_{self.current_epoch:05}.png"
            self.save_image(file_name, render["mask"][b_idx])

            file_name = f"mask/{f_idx:03}_{self.current_epoch:05}.png"
            self.save_image(file_name, mask[b_idx])

            file_name = f"render_depth/{f_idx:03}_{self.current_epoch:05}.png"
            self.save_image(file_name, render["depth_image"][b_idx])

            file_name = f"depth/{f_idx:03}_{self.current_epoch:05}.png"
            depth_image = renderer.depth_to_depth_image(renderer.point_to_depth(point))
            self.save_image(file_name, depth_image[b_idx])

            file_name = f"render_normal/{f_idx:03}_{self.current_epoch:05}.png"
            self.save_image(file_name, render["normal_image"][b_idx])

            file_name = f"normal/{f_idx:03}_{self.current_epoch:05}.png"
            normal_image = renderer.normal_to_normal_image(normal, mask)
            self.save_image(file_name, normal_image[b_idx])

            file_name = f"render_shading/{f_idx:03}_{self.current_epoch:05}.png"
            self.save_image(file_name, render["shading_image"][b_idx])

            file_name = f"render_full/{f_idx:03}_{self.current_epoch:05}.png"
            render_full = color.clone()
            color_mask = (
                (render["point"][:, :, :, 2] < point[:, :, :, 2])
                | (render["mask"] & ~mask)
            ) & render["mask"]
            render_full[color_mask] = render["shading_image"][color_mask]
            self.save_image(file_name, render_full[b_idx])

            # save the gt points
            file_name = f"point/{f_idx:03}_{self.current_epoch:05}.npy"
            points = point[mask].reshape(B, -1, 3)
            self.save_points(file_name, points[b_idx])

            # save the flame vertices
            file_name = f"render_point/{f_idx:03}_{self.current_epoch:05}.npy"
            render_points = render["point"][render["mask"]].reshape(B, -1, 3)
            self.save_points(file_name, render_points[b_idx])

        return loss

    def optimization_params(self):
        pose_params = [
            {"params": self.global_pose.parameters()},
            {"params": self.transl.parameters()},
        ]
        return {"params": pose_params, "lr": self.hparams["lr"]}
