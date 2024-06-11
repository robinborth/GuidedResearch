from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.loggers import Logger
from matplotlib import cm
from PIL import Image

from lib.model.flame import FLAME
from lib.model.loss import (
    calculate_point2plane,
    calculate_point2point,
    landmark_2d_distance,
    landmark_3d_distance,
)
from lib.renderer.renderer import Renderer


class FlameLogger:
    def __init__(self, logger: Logger, model: FLAME):
        self.logger = logger
        self.model = model

        self.save_dir = self.logger.save_dir
        self.current_epoch = 0
        self.max_loss = 1e-02

    def log(self, name: str, value: Any, step: None | int = None):
        self.logger.log_metrics({name: value}, step=step)

    @property
    def camera(self):
        return self.model.renderer.camera

    def iter_debug_idx(self, batch: dict):
        B = batch["frame_idx"].shape[0]
        for idx in range(B):
            yield B, idx, batch["frame_idx"][idx].item()

    def log_loss(self, batch: dict, model: dict):
        """The max is an error of 10cm"""
        # color entcoding for the loss map
        norm = plt.Normalize(0.0, vmax=self.max_loss)
        cmap = plt.get_cmap("jet")
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)

        for _, b_idx, f_idx in self.iter_debug_idx(batch):
            # point to point error map
            file_name = f"error_point_to_point/{f_idx:05}/{self.current_epoch:05}.png"
            loss = torch.sqrt(
                calculate_point2point(batch["point"], model["point"])
            )  # (B, W, H)
            error_map = loss[b_idx].detach().cpu().numpy()  # dist in m
            error_map = torch.from_numpy(sm.to_rgba(error_map)[:, :, :3])
            error_map[~model["r_mask"][b_idx], :] = 1.0
            error_map = (error_map * 255).to(torch.uint8)
            self.save_image(file_name, error_map)

            # point to plane error map
            file_name = f"error_point_to_plane/{f_idx:05}/{self.current_epoch:05}.png"
            loss = torch.sqrt(
                calculate_point2plane(
                    p=model["point"],
                    q=batch["point"],
                    n=model["normal"],
                )
            )  # (B, W, H)
            error_map = loss[b_idx].detach().cpu().numpy()  # dist in m
            error_map = torch.from_numpy(sm.to_rgba(error_map)[:, :, :3])
            error_map[~model["r_mask"][b_idx], :] = 1.0
            error_map = (error_map * 255).to(torch.uint8)
            self.save_image(file_name, error_map)

            # error mask
            file_name = f"error_mask/{f_idx:05}/{self.current_epoch:05}.png"
            self.save_image(file_name, model["mask"][b_idx])

    def log_render(self, batch: dict, model: dict):
        for _, b_idx, f_idx in self.iter_debug_idx(batch):
            file_name = f"render_mask/{f_idx:05}/{self.current_epoch:05}.png"
            self.save_image(file_name, model["r_mask"][b_idx])

            file_name = f"render_depth/{f_idx:05}/{self.current_epoch:05}.png"
            self.save_image(file_name, model["depth_image"][b_idx])

            file_name = f"render_normal/{f_idx:05}/{self.current_epoch:05}.png"
            self.save_image(file_name, model["normal_image"][b_idx])

            file_name = f"render_color/{f_idx:05}/{self.current_epoch:05}.png"
            self.save_image(file_name, model["color"][b_idx])

            file_name = f"render_merged_depth/{f_idx:05}/{self.current_epoch:05}.png"
            render_merged_depth = batch["color"].clone()
            color_mask = (
                (model["point"][:, :, :, 2] < batch["point"][:, :, :, 2])
                | (model["r_mask"] & ~batch["mask"])
            ) & model["r_mask"]
            render_merged_depth[color_mask] = model["color"][color_mask]
            self.save_image(file_name, render_merged_depth[b_idx])

            file_name = f"render_merged/{f_idx:05}/{self.current_epoch:05}.png"
            render_merged = batch["color"].clone()
            render_merged[model["r_mask"]] = model["color"][model["r_mask"]]
            self.save_image(file_name, render_merged[b_idx])

            file_name = f"render_landmark/{f_idx:05}/{self.current_epoch:05}.png"
            lm_2d_screen = self.camera.xy_ndc_to_screen(model["lm_2d_ndc"])
            x_idx = lm_2d_screen[b_idx, :, 0].to(torch.int32)
            y_idx = lm_2d_screen[b_idx, :, 1].to(torch.int32)
            color_image = model["color"][b_idx]
            color_image[y_idx, x_idx, :] = 255
            self.save_image(file_name, color_image)

    def log_3d_points(self, batch: dict, model: dict):
        for B, b_idx, f_idx in self.iter_debug_idx(batch):
            # save the gt points
            file_name = f"point_batch/{f_idx:05}/{self.current_epoch:05}.npy"
            points = batch["point"][b_idx][batch["mask"][b_idx]].reshape(-1, 3)
            self.save_points(file_name, points)
            # save the flame points
            file_name = f"point_render/{f_idx:05}/{self.current_epoch:05}.npy"
            render_points = model["point"][b_idx][model["r_mask"][b_idx]].reshape(-1, 3)
            self.save_points(file_name, render_points[b_idx])
            # save the lm_3d gt points
            file_name = f"point_batch_landmark/{f_idx:05}/{self.current_epoch:05}.npy"
            self.save_points(file_name, batch["lm_3d_camera"][b_idx])
            # save the lm_3d flame points
            file_name = f"point_render_landmark/{f_idx:05}/{self.current_epoch:05}.npy"
            self.save_points(file_name, model["lm_3d_camera"][b_idx])

    def log_input_batch(self, batch: dict, model: dict):
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
            lm_2d_screen = self.camera.xy_ndc_to_screen(batch["lm_2d_ndc"])
            x_idx = lm_2d_screen[b_idx, :, 0].to(torch.int32)
            y_idx = lm_2d_screen[b_idx, :, 1].to(torch.int32)
            color_image = batch["color"][b_idx]
            color_image[y_idx, x_idx, :] = 255
            self.save_image(file_name, color_image)

    def log_metrics(self, batch: dict, model: dict):
        # out of memory problems
        # chamfers = []
        # for _, b_idx, _ in self.iter_debug_idx(batch):
        #     # debug chamfer loss
        #     chamfer = chamfer_distance(
        #         render["point"][b_idx][render["mask"][b_idx]].reshape(1, -1, 3),
        #         batch["point"][b_idx][batch["mask"][b_idx]].reshape(1, -1, 3),
        #     )  # (B,)
        #     chamfers.append(chamfer)
        # self.log("train/chamfer", torch.stack(chamfers).mean(), prog_bar=True)
        # debug lm_3d loss, the dataset contains all >400 mediapipe landmarks
        lm_3d_camera = self.model.extract_landmarks(batch["lm_3d_camera"])
        lm_3d_dist = landmark_3d_distance(model["lm_3d_camera"], lm_3d_camera)  # (B,)
        self.log("train/lm_3d_loss", lm_3d_dist.mean())

        # debug lm_3d loss, the dataset contains all >400 mediapipe landmarks
        lm_2d_ndc = self.model.extract_landmarks(batch["lm_2d_ndc"])
        lm_2d_dist = landmark_2d_distance(model["lm_2d_ndc"], lm_2d_ndc)  # (B,)
        self.log("train/lm_2d_loss", lm_2d_dist.mean())

        # debug the overlap of the mask
        for key, value in model.items():
            if "mask" in key:
                self.log(f"debug/{key}", float(value.sum()))

    def log_params(self, batch: dict):
        for p_name in self.model.optimization_parameters:
            param = getattr(self, p_name, None)
            if p_name in batch and param is not None:
                if p_name in ["shape_params"]:
                    weight = param(batch["shape_idx"])
                else:
                    weight = param(batch["frame_idx"])
                param_dist = torch.norm(batch[p_name] - weight, dim=-1).mean()
                self.log(f"debug/param_{p_name}_l2", param_dist)

    def log_gradients(self, optimizer):
        for p_name in self.model.optimization_parameters:
            param = getattr(self, p_name, None)
            if param is None or not param.weight.requires_grad:
                continue
            for i in range(param.weight.shape[-1]):
                p = param.weight[:, i].mean()
                self.log(f"debug/{p_name}_{i}", p)
            for i in range(param.weight.grad.shape[-1]):
                p = param.weight.grad[:, i].mean()
                self.log(f"debug/{p_name}_{i}_grad", p)
            self.log(f"debug/{p_name}_mean", param.weight.mean())
            self.log(f"debug/{p_name}_absmax", param.weight.abs().max())
            self.log(f"debug/{p_name}_mean_grad", param.weight.grad.mean())
            self.log(f"debug/{p_name}_absmax_grad", param.weight.grad.abs().max())

    def save_image(self, file_name: str, image: torch.Tensor):
        path = Path(self.save_dir) / file_name  # type: ignore
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image.detach().cpu().numpy()).save(path)

    def save_points(self, file_name: str, points: torch.Tensor):
        path = Path(self.save_dir) / file_name  # type: ignore
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            np.save(f, points.detach().cpu().numpy())
