from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import yaml
from matplotlib import cm
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from lib.data.datamodule import DPHMDataModule
from lib.model.flame import FLAME
from lib.model.loss import (
    landmark_2d_distance,
    landmark_3d_distance,
    point2plane_distance,
    point2point_distance,
    regularization_distance,
)
from lib.rasterizer import Rasterizer
from lib.renderer.camera import Camera
from lib.renderer.renderer import Renderer
from lib.utils.logger import create_video


class FlameLogger:
    def __init__(
        self,
        save_dir: str,
        project: str,
        entity: str,
        group: str,
        tags: str,
        max_loss: float = 1e-02,
        mode: str = "",
    ):
        # settings
        self.save_dir = save_dir
        self.project = project
        self.entity = entity
        self.group = group
        self.tags = tags
        self.max_loss = max_loss
        # state
        self.iter_step = 0
        self.optim_step = 0
        self.global_step = 0
        self.mode = mode
        self.capture_video = False
        self.optimizer: Any = None

    def init_logger(self, model: FLAME, cfg: DictConfig):
        config: dict = OmegaConf.to_container(cfg, resolve=True)  # type: ignore
        self.logger = wandb.init(
            dir=self.save_dir,
            project=self.project,
            entity=self.entity,
            group=self.group,
            tags=self.tags,
            config=config,
        )
        self.model = model

    def log(self, name: str, value: Any, step: None | int = None, commit: bool = True):
        self.logger.log({name: value}, commit=commit)

    def log_dict(self, value: dict, step: None | int = None, commit: bool = True):
        self.logger.log(value, commit=commit)

    def log_path(self, folder: str, frame_idx: int, suffix: str):
        if self.capture_video:
            return f"final/{folder}/{frame_idx:05}.{suffix}"
        _f = f"{self.mode}/{folder}" if self.mode else folder
        return f"{_f}/{frame_idx:05}/{self.iter_step:05}.{suffix}"

    @property
    def camera(self):
        return self.model.renderer.camera

    def iter_debug_idx(self, batch: dict):
        B = batch["frame_idx"].shape[0]
        for idx in range(B):
            yield B, idx, batch["frame_idx"][idx].item()

    def log_error(self, batch: dict, model: dict):
        """The max is an error of 10cm"""
        # color entcoding for the loss map
        norm = plt.Normalize(0.0, vmax=self.max_loss)
        cmap = plt.get_cmap("jet")
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)

        for _, b_idx, f_idx in self.iter_debug_idx(batch):
            # point to point error map
            file_name = self.log_path("error_point_to_point", f_idx, "png")
            loss = point2point_distance(batch["point"], model["point"])  # (B, W, H)
            error_map = loss[b_idx].detach().cpu().numpy()  # dist in m
            error_map = torch.from_numpy(sm.to_rgba(error_map)[:, :, :3])
            error_map[~model["r_mask"][b_idx], :] = 1.0
            error_map = (error_map * 255).to(torch.uint8)
            self.save_image(file_name, error_map)

            # point to plane error map
            file_name = self.log_path("error_point_to_plane", f_idx, "png")
            loss = point2plane_distance(
                p=model["point"],
                q=batch["point"],
                n=model["normal"],
            )  # (B, W, H)
            error_map = loss[b_idx].detach().cpu().numpy()  # dist in m
            error_map = torch.from_numpy(sm.to_rgba(error_map)[:, :, :3])
            error_map[~model["r_mask"][b_idx], :] = 1.0
            error_map = (error_map * 255).to(torch.uint8)
            self.save_image(file_name, error_map)

            # error mask
            file_name = self.log_path("error_mask", f_idx, "png")
            self.save_image(file_name, model["mask"][b_idx])

            # normal distance
            file_name = self.log_path("error_normal", f_idx, "png")
            normal_dot = (batch["normal"][b_idx] * model["normal"][b_idx]).sum(-1)
            normal_map = normal_dot.detach().cpu().numpy()
            normal_map = (1 - ((normal_map + 1) / 2)) * self.max_loss
            normal_map = torch.from_numpy(sm.to_rgba(normal_map)[:, :, :3])
            normal_map[~model["r_mask"][b_idx], :] = 1.0
            normal_map = (normal_map * 255).to(torch.uint8)
            self.save_image(file_name, normal_map)

        # debug the overlap of the mask
        for key, value in model.items():
            if "mask" in key:
                self.log(f"mask/{key}", float(value.sum()))

    def log_render(self, batch: dict, model: dict):
        for _, b_idx, f_idx in self.iter_debug_idx(batch):
            file_name = self.log_path("render_mask", f_idx, "png")
            self.save_image(file_name, model["r_mask"][b_idx])

            file_name = self.log_path("render_depth", f_idx, "png")
            self.save_image(file_name, model["depth_image"][b_idx])

            file_name = self.log_path("render_normal", f_idx, "png")
            self.save_image(file_name, model["normal_image"][b_idx])

            file_name = self.log_path("render_color", f_idx, "png")
            self.save_image(file_name, model["color"][b_idx])

            file_name = self.log_path("render_merged_depth", f_idx, "png")
            render_merged_depth = batch["color"].clone()
            color_mask = (
                (model["point"][:, :, :, 2] < batch["point"][:, :, :, 2])
                | (model["r_mask"] & ~batch["mask"])
            ) & model["r_mask"]
            render_merged_depth[color_mask] = model["color"][color_mask]
            self.save_image(file_name, render_merged_depth[b_idx])

            file_name = self.log_path("render_merged", f_idx, "png")
            render_merged = batch["color"].clone()
            render_merged[model["r_mask"]] = model["color"][model["r_mask"]]
            self.save_image(file_name, render_merged[b_idx])

            file_name = self.log_path("render_landmark", f_idx, "png")
            lm_2d_screen = self.camera.xy_ndc_to_screen(model["lm_2d_ndc"])
            x_idx = lm_2d_screen[b_idx, :, 0].to(torch.int32)
            y_idx = lm_2d_screen[b_idx, :, 1].to(torch.int32)
            color_image = model["color"][b_idx]
            color_image[y_idx, x_idx, :] = 255
            self.save_image(file_name, color_image)

    def log_3d_points(self, batch: dict, model: dict):
        for B, b_idx, f_idx in self.iter_debug_idx(batch):
            # save the gt points
            file_name = self.log_path("point_batch", f_idx, "npy")
            points = batch["point"][b_idx][batch["mask"][b_idx]].reshape(-1, 3)
            self.save_points(file_name, points)
            # save the flame points
            file_name = self.log_path("point_render", f_idx, "npy")
            render_points = model["point"][b_idx][model["r_mask"][b_idx]].reshape(-1, 3)
            self.save_points(file_name, render_points[b_idx])
            # save the lm_3d gt points
            file_name = self.log_path("point_batch_landmark", f_idx, "npy")
            self.save_points(file_name, batch["lm_3d_camera"][b_idx])
            # save the lm_3d flame points
            file_name = self.log_path("point_render_landmark", f_idx, "npy")
            self.save_points(file_name, model["lm_3d_camera"][b_idx])

    def log_input_batch(self, batch: dict, model: dict):
        for _, b_idx, f_idx in self.iter_debug_idx(batch):
            file_name = self.log_path("batch_mask", f_idx, "png")
            self.save_image(file_name, batch["mask"][b_idx])

            file_name = self.log_path("batch_depth", f_idx, "png")
            depth = Renderer.point_to_depth(batch["point"])
            depth_image = Renderer.depth_to_depth_image(depth)
            self.save_image(file_name, depth_image[b_idx])

            file_name = self.log_path("batch_normal", f_idx, "png")
            normal_image = Renderer.normal_to_normal_image(
                batch["normal"], batch["mask"]
            )
            self.save_image(file_name, normal_image[b_idx])

            file_name = self.log_path("batch_color", f_idx, "png")
            self.save_image(file_name, batch["color"][b_idx])

            file_name = self.log_path("batch_landmark", f_idx, "png")
            lm_2d_screen = self.camera.xy_ndc_to_screen(batch["lm_2d_ndc"])
            x_idx = lm_2d_screen[b_idx, :, 0].to(torch.int32)
            y_idx = lm_2d_screen[b_idx, :, 1].to(torch.int32)
            color_image = batch["color"][b_idx]
            color_image[y_idx, x_idx, :] = 255
            self.save_image(file_name, color_image)

    def log_metrics(self, batch, model: dict, verbose: bool = False):
        m = model
        mask = m["mask"]

        point2plane = point2plane_distance(m["point"], m["point_gt"], m["normal"])
        self.log(f"{self.mode}/metric/point2plane", point2plane.mean())

        point2point = point2point_distance(m["point"], m["point_gt"])
        self.log(f"{self.mode}/metric/point2point", point2point.mean())

        lm_3d_dist = landmark_3d_distance(m["lm_3d_camera"], m["lm_3d_camera_gt"])
        self.log(f"{self.mode}/metric/landmark_3d", lm_3d_dist.mean())

        lm_2d_dist = landmark_3d_distance(m["lm_2d_ndc"], m["lm_2d_ndc_gt"])
        self.log(f"{self.mode}/metric/landmark_2d", lm_2d_dist.mean())

        # debug the regularization
        params = ["expression_params", "shape_params"]
        reg_dist = regularization_distance([m[p] for p in params])
        self.log(f"{self.mode}/metric/regularization", reg_dist.mean())

        # debug for each frame own loss curve
        if self.mode == "sequential" and verbose:
            i = 0
            for _, b_idx, f_idx in self.iter_debug_idx(batch):
                pre = f"{self.mode}/metric/{f_idx:03}"
                j = mask[: b_idx + 1].sum()
                self.log(f"{pre}/point2plane", point2plane[i:j].mean())
                self.log(f"{pre}/point2point", point2point[i:j].mean())
                self.log(f"{pre}/landmark_3d", lm_3d_dist[b_idx].mean())
                self.log(f"{pre}/landmark_2d", lm_2d_dist[b_idx].mean())
                i = j

    def log_loss(self, loss_info: dict[str, torch.Tensor]):
        for key, value in loss_info.items():
            k = f"loss/{key}" if key != "loss" else "loss/"
            v = value.mean() if value.numel() else torch.tensor(0.0)
            self.log(f"{self.mode}/{k}", v)
        return loss_info["loss"].mean().item()

    def log_params(self, batch: dict):
        for _, _, f_idx in self.iter_debug_idx(batch):
            params = {}
            for p_name in self.model.frame_p_names:
                module = getattr(self.model, p_name)
                weight = module.weight[f_idx].detach().cpu().numpy()
                params[p_name] = [float(w) for w in weight]
            for p_name in self.model.shape_p_names:
                module = getattr(self.model, p_name)
                weight = module.weight[0].detach().cpu().numpy()
                params[p_name] = [float(w) for w in weight]
            file_name = self.log_path("model_params", f_idx, "yaml")
            params = dict(sorted(params.items(), key=lambda item: len(item[1])))
            self.save_params(file_name, params)

    def log_gradients(self, verbose: bool = False):
        log = {}
        for p_name, weight in zip(self.optimizer._p_names, self.optimizer._params):
            grad = weight.grad

            log[f"{self.mode}/{p_name}/weight/mean"] = weight.mean()
            log[f"{self.mode}/{p_name}/weight/absmax"] = weight.abs().max()
            log[f"{self.mode}/{p_name}/grad/mean"] = grad.mean()
            log[f"{self.mode}/{p_name}/grad/absmax"] = grad.abs().max()

            if verbose:
                for i in range(weight.shape[-1]):
                    value = weight[:, i].mean()
                    log[f"{self.mode}/{p_name}/{i:03}/weight"] = value
                for i in range(grad.shape[-1]):
                    value = grad[:, i].mean()
                    log[f"{self.mode}/{p_name}/{i:03}/grad"] = value

        damping_factor = getattr(self.optimizer, "damping_factor", None)
        if damping_factor:
            log[f"{self.mode}/optimizer/damping_factor"] = damping_factor

        self.log_dict(log)

    def save_image(self, file_name: str, image: torch.Tensor):
        path = Path(self.save_dir) / file_name  # type: ignore
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image.detach().cpu().numpy()).save(path)

    def save_points(self, file_name: str, points: torch.Tensor):
        path = Path(self.save_dir) / file_name  # type: ignore
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            np.save(f, points.detach().cpu().numpy())

    def save_params(self, file_name: str, params: dict):
        path = Path(self.save_dir) / file_name  # type: ignore
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            out = yaml.dump(params, sort_keys=False)
            f.write(out)

    def capture_screen(
        self,
        camera: Camera,
        rasterizer: Rasterizer,
        datamodule: DPHMDataModule,
        model: FLAME,
        idx: int,
    ):
        # set state
        prev_iter_step = self.iter_step
        prev_scale = camera.scale
        self.capture_video = True
        self.iter_step = 0
        camera.update(scale=1)
        rasterizer.update(width=camera.width, height=camera.height)
        datamodule.update_dataset(camera=camera, rasterizer=rasterizer)

        # inference
        datamodule.update_idxs([idx])
        batch = datamodule.fetch()
        with torch.no_grad():
            out = model.correspondence_step(batch)

        # log
        self.log_render(batch=batch, model=out)
        self.log_input_batch(batch=batch, model=out)
        self.log_error(batch=batch, model=out)

        # restore state
        camera.update(scale=prev_scale)
        rasterizer.update(width=camera.width, height=camera.height)
        datamodule.update_dataset(camera=camera, rasterizer=rasterizer)
        self.capture_video = False
        self.iter_step = prev_iter_step

    def log_video(self, name: str, framerate: int = 30):
        _video_dir = Path(self.save_dir) / "final" / name
        assert _video_dir.exists()
        video_dir = str(_video_dir.resolve())
        video_path = str((Path(self.save_dir) / "video" / f"{name}.mp4").resolve())
        create_video(video_dir=video_dir, video_path=video_path, framerate=framerate)
