from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from lib.model.lbs import lbs
from lib.model.loss import chamfer_distance_loss, landmark_3d_loss, point_to_point_loss
from lib.model.utils import load_flame, load_flame_masks, load_static_landmark_embedding
from lib.renderer.renderer import Renderer
from lib.utils.logger import create_logger

log = create_logger("flame")


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
        lm_weight: float = 1.0,
        p2p_weight: float = 0.1,
        optimize_frames: int = 1,
        # debug settings
        renderer: Renderer | None = None,
        save_interval: int = 50,
    ):

        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["renderer"])

        # load the face model
        flame_model = load_flame(flame_dir=flame_dir, return_tensors="pt")

        # load the faces, mean vertices and pca bases
        self.faces = nn.Parameter(flame_model["f"], requires_grad=False)  # (9976, 3)
        self.v_template = nn.Parameter(flame_model["v_template"])  # (5023, 3)
        self.shapedirs = nn.Parameter(flame_model["shapedirs"])  # (5023, 3, 400)

        # shape parameters
        zero_shape = torch.zeros(300 - num_shape_params)
        self.zero_shape = nn.Parameter(zero_shape)
        shape_params = torch.zeros(num_shape_params)
        self.shape_params = nn.Parameter(shape_params)

        # expression parameters
        zero_expression = torch.zeros(100 - num_expression_params)
        self.zero_expression = nn.Parameter(zero_expression)
        expression_params = torch.zeros(num_expression_params)
        self.expression_params = nn.Parameter(expression_params)

        # pose parameters
        self.global_pose = nn.Parameter(torch.tensor([torch.pi, 0.0, 0.0]))
        self.neck_pose = nn.Parameter(torch.zeros(3))
        self.jaw_pose = nn.Parameter(torch.zeros(3))
        self.eye_pose = nn.Parameter(torch.zeros(6))

        # translation and scale pose
        self.transl = nn.Parameter(torch.zeros(3))
        self.scale = nn.Parameter(torch.ones(3))

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
        face_mask = load_flame_masks(flame_dir, return_tensors="pt")["face"]
        self.face_vertices_mask = torch.nn.Parameter(face_mask, requires_grad=False)

        # flame landmarks to guide sparse landmark loss
        lms = load_static_landmark_embedding(flame_dir, return_tensors="pt")
        lm_faces = self.faces[lms["lm_face_idx"]]  # (105, )
        self.lm_faces = torch.nn.Parameter(lm_faces, requires_grad=False)
        lm_bary_coords = lms["lm_bary_coords"]  # (105, 3)
        self.lm_bary_coords = torch.nn.Parameter(lm_bary_coords, requires_grad=False)
        lm_idx = lms["lm_mediapipe_idx"]  # (105,)
        self.lm_mediapipe_idx = torch.nn.Parameter(lm_idx, requires_grad=False)

        # visualization/debugging
        if renderer is None:
            log.info("FLAME model does not have a renderer specified!")
        self.renderer = renderer

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

        TODO: Output 3D facial landmarks

        Args:
            shape_params (torch.tensor): self.hparams["shape_params"]
            expression_params (torch.tensor): self.hparams["expression_params"]
            global_pose (torch.tensor): number of global pose parameters (3,)
            neck_pose (torch.tensor): number of neck pose parameters (3,)
            jaw_pose (torch.tensor): number of jaw pose parameters (3,)
            eye_pose (torch.tensor): number of eye pose parameters (6,)
            transl_pose (torch.tensor): number of eye pose parameters (3,)

        Return:
            (torch.Tensor): The mesh vertices of dim (V, 3)
        """
        # create the shape parameters
        shape_params = shape_params if shape_params is not None else self.shape_params
        shape = torch.cat([shape_params, self.zero_shape])  # (300, )
        # create the expression parameters
        expression_params = (
            expression_params
            if expression_params is not None
            else self.expression_params
        )
        expression = torch.cat([expression_params, self.zero_expression])  # (100, )
        # create the betas merged with shape and expression
        betas = torch.cat([shape, expression])

        # create the pose merged with global, jaw, neck and left/right eye
        global_pose = global_pose if global_pose is not None else self.global_pose
        jaw_pose = jaw_pose if jaw_pose is not None else self.jaw_pose
        neck_pose = neck_pose if neck_pose is not None else self.neck_pose
        eye_pose = eye_pose if eye_pose is not None else self.eye_pose
        pose = torch.cat([global_pose, neck_pose, jaw_pose, eye_pose], dim=-1)  # (15,)

        # the translation and scaling of the vertices in world space
        transl = transl if transl is not None else self.transl
        scale = scale if scale is not None else self.scale

        # apply the linear blend skinning model
        vertices, _ = lbs(
            betas.unsqueeze(0),
            pose.unsqueeze(0),
            self.v_template.unsqueeze(0),
            # weights/settings of the lbs model
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
        )
        vertices = vertices[0]

        # apply the translation and the scaling
        vertices += transl
        vertices *= scale

        return vertices

    def training_step(self, batch, batch_idx):
        points = batch["points"]  # (B, P, 3)
        mediapipe_lm3d = batch["mediapipe_lm3d"]  # (B, 478, 3)
        frame_idx = batch["frame_idx"]  # (B, )
        image = batch["image"]  # (B, H, W, 3)

        # apply the flame model to get the vertices in world space
        vertices = self.forward()  # (V, 3)

        p2p_loss = point_to_point_loss(
            vertices=vertices.unsqueeze(0),
            points=points,
        )
        p2p_loss = p2p_loss.mean()
        self.log("train/p2p_loss", p2p_loss, prog_bar=True)

        p2p_face_loss = point_to_point_loss(
            vertices=vertices.unsqueeze(0),
            points=points,
        )[self.face_vertices_mask]
        p2p_face_loss = p2p_face_loss.mean()
        self.log("train/p2p_face_loss", p2p_loss, prog_bar=True)

        cd_loss = chamfer_distance_loss(
            vertices=vertices.unsqueeze(0),
            points=points,
        ).mean()
        self.log("train/chamfer_distance_loss", cd_loss, prog_bar=True)

        lm_loss, vertices_lm3d = landmark_3d_loss(
            vertices=vertices,
            faces=self.lm_faces,
            bary_coords=self.lm_bary_coords,
            lm_mediapipe_idx=self.lm_mediapipe_idx,
            mediapipe_lm3d=mediapipe_lm3d[0],
        )
        lm_loss = lm_loss.mean()
        self.log("train/lm_loss", lm_loss, prog_bar=True)

        loss = p2p_face_loss + lm_loss
        self.log("train/loss", loss, prog_bar=True)

        # visualize the image that is optimized and save to disk and the 3d points
        if self.current_epoch + 1 % self.hparams["save_interval"] == 0:
            flame_image = self.renderer.render_color_image(
                vertices=vertices.detach().cpu(),
                faces=self.faces.detach().cpu(),
                image=image[0].detach().cpu(),  # (H, W, 3)
                vertices_mask=self.face_vertices_mask,
            )
            file_name = f"image/{frame_idx:03}_{self.current_epoch:05}.png"
            self.save_image(file_name, flame_image)
            file_name = f"pcd_vertices/{frame_idx:03}_{self.current_epoch:05}.npz"
            self.save_points(file_name, vertices[self.face_vertices_mask])
            file_name = f"pcd_points/{frame_idx:03}_{self.current_epoch:05}.npz"
            self.save_points(file_name, points[0])
            file_name = f"lm3d_vertices/{frame_idx:03}_{self.current_epoch:05}.npz"
            self.save_points(file_name, vertices_lm3d)
            file_name = f"lm3d_points/{frame_idx:03}_{self.current_epoch:05}.npz"
            self.save_points(file_name, mediapipe_lm3d[self.lm_mediapipe_idx])

        return loss

    def configure_optimizers(self):
        pose_params = [self.global_pose, self.transl]
        optimizer = torch.optim.Adam(pose_params, lr=self.hparams["lr"])
        if self.hparams["scheduler"] is not None:
            scheduler = self.hparams["scheduler"](optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "train/loss"},
            }
        return {"optimizer": optimizer}

    def save_image(self, file_name: str, image: torch.Tensor):
        path = Path(self.logger.save_dir) / file_name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image.detach().cpu().numpy()).save(path)

    def save_points(self, file_name: str, points: torch.Tensor):
        path = Path(self.logger.save_dir) / file_name
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            np.save(f, points.detach().cpu().numpy())
