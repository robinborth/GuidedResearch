from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from lib.model.lbs import lbs
from lib.model.loss import distance, landmark_3d_loss
from lib.model.utils import (
    bary_coord_interpolation,
    load_flame,
    load_flame_masks,
    load_static_landmark_embedding,
)
from lib.renderer.renderer import Renderer
from lib.utils.logger import create_logger

log = create_logger("flame")


# TODO good camera initialization, e.g. in the first frame we can already see the flame model
# TODO 2D depth image with 3d coordinates from the dataset
# TODO


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
        optimize_frames: int = 1,  # the number of different frames per shape
        optimize_shapes: int = 1,  # the number of different shapes (allways 1)
        # debug settings
        renderer: Renderer | None = None,
        save_interval: int = 50,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["renderer"])

        # load the face model
        flame_model = load_flame(flame_dir=flame_dir, return_tensors="pt")

        # shape parameters, default just 1 shape per optimization
        zero_shape = torch.zeros(300 - num_shape_params)
        self.zero_shape = nn.Parameter(zero_shape)
        shape_params = torch.zeros(optimize_shapes, num_shape_params)
        self.shape_params = self._embedding_table(shape_params)

        # expression parameters
        zero_expression = torch.zeros(100 - num_expression_params)
        self.zero_expression = nn.Parameter(zero_expression)
        expression_params = torch.zeros(optimize_frames, num_expression_params)
        self.expression_params = self._embedding_table(expression_params)

        # pose parameters
        global_pose = torch.tensor([[torch.pi, 0.0, 0.0]] * optimize_frames)
        self.global_pose = self._embedding_table(global_pose)
        self.neck_pose = self._embedding_table(torch.zeros(optimize_frames, 3))
        self.jaw_pose = self._embedding_table(torch.zeros(optimize_frames, 3))
        self.eye_pose = self._embedding_table(torch.zeros(optimize_frames, 6))

        # translation and scale pose
        self.transl = self._embedding_table(torch.zeros(optimize_frames, 3))
        self.scale = self._embedding_table(torch.ones(optimize_frames, 3))

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

    def _embedding_table(self, tensor: torch.Tensor):
        num_embeddings, embedding_dim = tensor.shape
        return nn.Embedding(num_embeddings, embedding_dim, _weight=tensor)

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

        landmarks = bary_coord_interpolation(
            faces=self.lm_faces,
            bary_coords=self.lm_bary_coords.expand(B, -1, -1),
            attributes=vertices,
        )  # (B, 105, 3)

        return vertices, landmarks  # (B, V, 3), (B, 105, 3)

    def training_step(self, batch, batch_idx):
        frame_idx = batch["frame_idx"]  # (B, )
        shape_idx = batch["shape_idx"]  # (B, )
        points = batch["points"]  # (B, P, 3)
        mediapipe_lm3d = batch["mediapipe_lm3d"]  # (B, 478, 3)
        image = batch["image"]  # (B, H', W', 3) scaled

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

        depth_image = self.renderer.render_depth(
            vertices=vertices,  # (B, V, 3)
            faces=self.faces,  # (F, 3)
        )  # (B, H, W)

        dist = distance(vertices=vertices, points=points)  # (B, V, P)
        p2p_loss = dist.min(-1).values.mean()  # (B, V)
        self.log("train/p2p_loss", p2p_loss, prog_bar=True)

        p2p_face_loss = dist.min(-1).values[:, self.face_vertices_mask]
        p2p_face_loss = p2p_face_loss.mean()  # (B', V)
        self.log("train/p2p_face_loss", p2p_loss, prog_bar=True)

        cd_loss = dist.min(-1).values.mean() + dist.min(-2).values.mean()  # (B,)
        self.log("train/chamfer_distance_loss", cd_loss, prog_bar=True)

        gt_landmarks = mediapipe_lm3d[:, self.lm_mediapipe_idx]
        lm_loss = landmark_3d_loss(landmarks, gt_landmarks)  # (105,)
        lm_loss = lm_loss.mean()
        self.log("train/lm_loss", lm_loss, prog_bar=True)

        loss = p2p_face_loss + lm_loss
        self.log("train/loss", loss, prog_bar=True)

        # visualize the image that is optimized and save to disk and the 3d points
        if (self.current_epoch + 0) % self.hparams["save_interval"] == 0:
            flame_image = self.renderer.render_depth(
                vertices=vertices,  # (B, V, 3)
                faces=self.faces,  # (F, 3)
                # image=image.detach().cpu(),  # (B, H, W, 3)
                # vertices_mask=self.face_vertices_mask,
            )  # (B, H, W)
            for frame_idx, flame_image in zip(frame_idx, flame_image):
                file_name = f"image/{frame_idx.item():03}_{self.current_epoch:05}.png"
                self.save_image(file_name, flame_image)
            # file_name = f"pcd_vertices/{frame_idx:03}_{self.current_epoch:05}.npz"
            # self.save_points(file_name, vertices[self.face_vertices_mask])
            # file_name = f"pcd_points/{frame_idx:03}_{self.current_epoch:05}.npz"
            # self.save_points(file_name, points[0])
            # file_name = f"lm3d_vertices/{frame_idx:03}_{self.current_epoch:05}.npz"
            # self.save_points(file_name, landmarks)
            # file_name = f"lm3d_points/{frame_idx:03}_{self.current_epoch:05}.npz"
            # self.save_points(file_name, gt_landmarks)

        return loss

    def configure_optimizers(self):
        pose_params = [
            {"params": self.global_pose.parameters()},
            {"params": self.transl.parameters()},
        ]
        optimizer = torch.optim.Adam(pose_params, lr=self.hparams["lr"])
        if self.hparams["scheduler"] is not None:
            scheduler = self.hparams["scheduler"](optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "train/loss"},
            }
        return {"optimizer": optimizer}

    def save_image(self, file_name: str, image: torch.Tensor):
        path = Path(self.logger.save_dir) / file_name  # type: ignore
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image.detach().cpu().numpy()).save(path)

    def save_points(self, file_name: str, points: torch.Tensor):
        path = Path(self.logger.save_dir) / file_name  # type: ignore
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            np.save(f, points.detach().cpu().numpy())
