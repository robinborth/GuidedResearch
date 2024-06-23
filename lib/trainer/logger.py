from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from matplotlib import cm
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch import profiler

from lib.model.flame import FLAME
from lib.model.loss import (
    calculate_point2plane,
    calculate_point2point,
    landmark_2d_distance,
    landmark_3d_distance,
)
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
        # self.logger.log({name: value}, step=self.global_step, commit=commit)
        self.logger.log({name: value}, commit=commit)

    def log_dict(self, value: dict, step: None | int = None, commit: bool = True):
        self.logger.log(value, commit=commit)

    def create_profiler(self):
        # TODO need to handle step and stop
        prof = profiler.profile(
            activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        prof.start()

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
            file_name = f"error_point_to_point/{f_idx:05}/{self.iter_step:05}.png"
            loss = torch.sqrt(
                calculate_point2point(batch["point"], model["point"])
            )  # (B, W, H)
            error_map = loss[b_idx].detach().cpu().numpy()  # dist in m
            error_map = torch.from_numpy(sm.to_rgba(error_map)[:, :, :3])
            error_map[~model["r_mask"][b_idx], :] = 1.0
            error_map = (error_map * 255).to(torch.uint8)
            self.save_image(file_name, error_map)

            # point to plane error map
            file_name = f"error_point_to_plane/{f_idx:05}/{self.iter_step:05}.png"
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
            file_name = f"error_mask/{f_idx:05}/{self.iter_step:05}.png"
            self.save_image(file_name, model["mask"][b_idx])

    def log_render(self, batch: dict, model: dict):
        for _, b_idx, f_idx in self.iter_debug_idx(batch):
            file_name = f"render_mask/{f_idx:05}/{self.iter_step:05}.png"
            self.save_image(file_name, model["r_mask"][b_idx])

            file_name = f"render_depth/{f_idx:05}/{self.iter_step:05}.png"
            self.save_image(file_name, model["depth_image"][b_idx])

            file_name = f"render_normal/{f_idx:05}/{self.iter_step:05}.png"
            self.save_image(file_name, model["normal_image"][b_idx])

            file_name = f"render_color/{f_idx:05}/{self.iter_step:05}.png"
            self.save_image(file_name, model["color"][b_idx])

            file_name = f"render_merged_depth/{f_idx:05}/{self.iter_step:05}.png"
            render_merged_depth = batch["color"].clone()
            color_mask = (
                (model["point"][:, :, :, 2] < batch["point"][:, :, :, 2])
                | (model["r_mask"] & ~batch["mask"])
            ) & model["r_mask"]
            render_merged_depth[color_mask] = model["color"][color_mask]
            self.save_image(file_name, render_merged_depth[b_idx])

            file_name = f"render_merged/{f_idx:05}/{self.iter_step:05}.png"
            render_merged = batch["color"].clone()
            render_merged[model["r_mask"]] = model["color"][model["r_mask"]]
            self.save_image(file_name, render_merged[b_idx])

            file_name = f"render_landmark/{f_idx:05}/{self.iter_step:05}.png"
            lm_2d_screen = self.camera.xy_ndc_to_screen(model["lm_2d_ndc"])
            x_idx = lm_2d_screen[b_idx, :, 0].to(torch.int32)
            y_idx = lm_2d_screen[b_idx, :, 1].to(torch.int32)
            color_image = model["color"][b_idx]
            color_image[y_idx, x_idx, :] = 255
            self.save_image(file_name, color_image)

    def log_3d_points(self, batch: dict, model: dict):
        for B, b_idx, f_idx in self.iter_debug_idx(batch):
            # save the gt points
            file_name = f"point_batch/{f_idx:05}/{self.iter_step:05}.npy"
            points = batch["point"][b_idx][batch["mask"][b_idx]].reshape(-1, 3)
            self.save_points(file_name, points)
            # save the flame points
            file_name = f"point_render/{f_idx:05}/{self.iter_step:05}.npy"
            render_points = model["point"][b_idx][model["r_mask"][b_idx]].reshape(-1, 3)
            self.save_points(file_name, render_points[b_idx])
            # save the lm_3d gt points
            file_name = f"point_batch_landmark/{f_idx:05}/{self.iter_step:05}.npy"
            self.save_points(file_name, batch["lm_3d_camera"][b_idx])
            # save the lm_3d flame points
            file_name = f"point_render_landmark/{f_idx:05}/{self.iter_step:05}.npy"
            self.save_points(file_name, model["lm_3d_camera"][b_idx])

    def log_input_batch(self, batch: dict, model: dict):
        for _, b_idx, f_idx in self.iter_debug_idx(batch):
            file_name = f"batch_mask/{f_idx:05}/{self.iter_step:05}.png"
            self.save_image(file_name, batch["mask"][b_idx])

            file_name = f"batch_depth/{f_idx:05}/{self.iter_step:05}.png"
            depth = Renderer.point_to_depth(batch["point"])
            depth_image = Renderer.depth_to_depth_image(depth)
            self.save_image(file_name, depth_image[b_idx])

            file_name = f"batch_normal/{f_idx:05}/{self.iter_step:05}.png"
            normal_image = Renderer.normal_to_normal_image(
                batch["normal"], batch["mask"]
            )
            self.save_image(file_name, normal_image[b_idx])

            file_name = f"batch_color/{f_idx:05}/{self.iter_step:05}.png"
            self.save_image(file_name, batch["color"][b_idx])

            file_name = f"batch_landmark/{f_idx:05}/{self.iter_step:05}.png"
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
        self.log("loss/landmark_3d", lm_3d_dist.mean())

        # debug lm_3d loss, the dataset contains all >400 mediapipe landmarks
        lm_2d_ndc = self.model.extract_landmarks(batch["lm_2d_ndc"])
        lm_2d_dist = landmark_2d_distance(model["lm_2d_ndc"], lm_2d_ndc)  # (B,)
        self.log("loss/landmark_2d", lm_2d_dist.mean())

        # debug the overlap of the mask
        for key, value in model.items():
            if "mask" in key:
                self.log(f"mask/{key}", float(value.sum()))

    def log_params(self, batch: dict):
        for p_name in self.model.full_p_names:
            param = getattr(self, p_name, None)
            if p_name in batch and param is not None:
                if p_name in ["shape_params"]:
                    weight = param(batch["shape_idx"])
                else:
                    weight = param(batch["frame_idx"])
                param_dist = torch.norm(batch[p_name] - weight, dim=-1).mean()
                self.log(f"debug/param_{p_name}_l2", param_dist)

    def log_gradients(self, optimizer):
        log = {}
        step = f"{self.iter_step:03}"
        for group in optimizer.param_groups:
            p_name = group["p_name"]
            weight = group["params"][0]
            grad = weight.grad

            for i in range(weight.shape[-1]):
                value = weight[:, i].mean()
                # log[f"weight/{step}/{p_name}_{i}"] = value
                log[f"weight/{p_name}_{i}_mean"] = value

            for i in range(grad.shape[-1]):
                value = grad[:, i].mean()
                # log[f"grad/{step}/{p_name}_{i}"] = value
                log[f"grad/{p_name}_{i}_mean"] = value

            value = weight.mean()
            log[f"weight/{step}/{p_name}_mean"] = value
            log[f"weight/{p_name}_mean"] = value

            value = weight.abs().max()
            log[f"weight/{step}/{p_name}_absmax"] = value
            log[f"weight/{p_name}_absmax"] = value

            value = grad.mean()
            log[f"grad/{step}/{p_name}_mean"] = value
            log[f"grad/{p_name}_mean"] = value

            value = grad.abs().max()
            log[f"grad/{step}/{p_name}_absmax"] = value
            log[f"grad/{p_name}_absmax"] = value

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

    def _flat_list_str(self, items):
        lstr = []
        for i in items:
            if isinstance(i, list):
                lstr.append(self._flat_list_str(i))
            else:
                lstr.append(str(i))
        return f"[{', '.join(lstr)}]"

    def _flat_hyparams(self, hparams: dict):
        params = {}
        for key, item in hparams.items():
            if isinstance(item, dict):
                for k, i in self._flat_hyparams(item).items():
                    params[f"{key}/{k}"] = i
            elif isinstance(item, list):
                params[key] = self._flat_list_str(item)
            else:
                params[key] = item
        return params

    def capture_screen(self, datamodule, model):
        batch = datamodule.fetch()
        with torch.no_grad():
            out = model.correspondence_step(batch)
        self.log_render(batch=batch, model=out)
        self.log_input_batch(batch=batch, model=out)
        self.log_loss(batch=batch, model=out)

    def log_video(self, name: str, framerate: int = 30):
        _video_dir = Path(self.save_dir) / name
        assert _video_dir.exists()
        video_dir = str(_video_dir.resolve())
        video_path = str((Path(self.save_dir) / "video" / f"{name}.mp4").resolve())
        create_video(video_dir=video_dir, video_path=video_path, framerate=framerate)
