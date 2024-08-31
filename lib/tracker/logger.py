from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from lightning.pytorch.loggers.wandb import WandbLogger
from matplotlib import cm
from omegaconf import DictConfig, OmegaConf
from PIL import Image

import wandb
from lib.data.datamodule import DPHMDataModule
from lib.model.flame import Flame
from lib.rasterizer import Rasterizer
from lib.renderer.camera import Camera
from lib.renderer.renderer import Renderer
from lib.tracker.timer import TimeTracker
from lib.utils.distance import (
    landmark_3d_distance,
    point2plane_distance,
    point2point_distance,
    regularization_distance,
)
from lib.utils.video import create_video
from lib.utils.visualize import (
    change_color,
    visualize_depth_merged,
    visualize_grid,
    visualize_merged,
    visualize_normal_error,
    visualize_point2plane_error,
    visualize_point2point_error,
)


class FlameLogger(WandbLogger):
    def __init__(self, mode: str = "", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.mode = mode
        self.iter_step = 0
        self.capture_eval = False

    def log_step(
        self,
        batch: dict,
        out: dict,
        flame: Flame,
        renderer: Renderer,
        mode: str = "train",
        batch_idx: int = 0,
    ):
        # update full resolution
        gt_out = flame.render(renderer=renderer, params=batch["params"])
        init_out = flame.render(renderer=renderer, params=batch["init_params"])
        new_out = flame.render(renderer=renderer, params=out["params"])

        wandb_images = []

        # initial color image
        visualize_grid(images=batch["init_color"])
        wandb_images.append(wandb.Image(plt))
        plt.close()

        # target color image
        visualize_grid(images=batch["color"])
        wandb_images.append(wandb.Image(plt))
        plt.close()

        # target normal image
        normal_image = Renderer.normal_to_normal_image(batch["normal"], batch["mask"])
        visualize_grid(images=normal_image)
        wandb_images.append(wandb.Image(plt))
        plt.close()

        # point clouds merged: initial - target
        images = visualize_depth_merged(
            s_color=change_color(gt_out["color"], gt_out["mask"], code=0),
            s_point=gt_out["point"],
            s_mask=gt_out["mask"],
            t_color=change_color(init_out["color"], init_out["mask"], code=1),
            t_point=init_out["point"],
            t_mask=init_out["mask"],
        )
        visualize_grid(images=images)
        wandb_images.append(wandb.Image(plt))
        plt.close()

        # point clouds merged: new - target
        images = visualize_depth_merged(
            s_color=change_color(gt_out["color"], gt_out["mask"], code=0),
            s_point=gt_out["point"],
            s_mask=gt_out["mask"],
            t_color=change_color(new_out["color"], new_out["mask"], code=2),
            t_point=new_out["point"],
            t_mask=new_out["mask"],
        )
        visualize_grid(images)
        wandb_images.append(wandb.Image(plt))
        plt.close()

        # flame init
        images = change_color(color=init_out["color"], mask=init_out["mask"], code=1)
        visualize_grid(images=images)
        wandb_images.append(wandb.Image(plt))
        plt.close()

        # flame new
        images = change_color(color=new_out["color"], mask=new_out["mask"], code=2)
        visualize_grid(images=images)
        wandb_images.append(wandb.Image(plt))
        plt.close()

        # flame target
        images = change_color(color=gt_out["color"], mask=gt_out["mask"], code=0)
        visualize_grid(images=images)
        wandb_images.append(wandb.Image(plt))
        plt.close()

        # weights map
        for weight in out["weights"]:
            visualize_grid(images=weight)
            wandb_images.append(wandb.Image(plt))
            plt.close()

        # init error map
        images = visualize_point2plane_error(
            s_point=batch["point"][batch_idx],
            t_normal=init_out["normal"][batch_idx],
            t_point=init_out["point"][batch_idx],
            t_mask=init_out["mask"][batch_idx],
        )
        visualize_grid(images=images.unsqueeze(0))
        wandb_images.append(wandb.Image(plt, caption="init_error_map"))
        plt.close()

        # new error map
        images = visualize_point2plane_error(
            s_point=batch["point"][batch_idx],
            t_normal=new_out["normal"][batch_idx],
            t_point=new_out["point"][batch_idx],
            t_mask=new_out["mask"][batch_idx],
        )
        visualize_grid(images=images.unsqueeze(0))
        wandb_images.append(wandb.Image(plt, caption="new_error_map"))
        plt.close()

        # gt error map
        images = visualize_point2plane_error(
            s_point=batch["point"][batch_idx],
            t_normal=gt_out["normal"][batch_idx],
            t_point=gt_out["point"][batch_idx],
            t_mask=gt_out["mask"][batch_idx],
        )
        visualize_grid(images=images.unsqueeze(0))
        wandb_images.append(wandb.Image(plt, caption="gt_error_map"))
        plt.close()

        # reset scale to previous
        scale = renderer.camera.scale
        renderer.update(scale=2)
        gt_out = flame.render(renderer=renderer, params=batch["params"])
        init_out = flame.render(renderer=renderer, params=batch["init_params"])
        new_out = flame.render(renderer=renderer, params=out["params"])

        # flame init
        images = init_out["color"]
        visualize_grid(images=images)
        wandb_images.append(wandb.Image(plt))
        plt.close()

        # flame new
        images = new_out["color"]
        visualize_grid(images=images)
        wandb_images.append(wandb.Image(plt))
        plt.close()

        # flame target
        images = gt_out["color"]
        visualize_grid(images=images)
        wandb_images.append(wandb.Image(plt))
        plt.close()

        renderer.update(scale=scale)

        self.log_image(f"{mode}/images", wandb_images)  # type:ignore

    def log_params(
        self,
        name: str,
        params: dict,
        flame: Flame,
        renderer: Renderer,
        code: int = 0,
    ):
        out = flame.render(renderer=renderer, params=params)
        images = change_color(color=out["color"], mask=out["mask"], code=code)
        visualize_grid(images=images)
        img = wandb.Image(plt)
        plt.close()
        self.log_image(f"{self.mode}/{name}", [img])  # type: ignore

    def log_merged(
        self,
        name: str,
        params: dict,
        flame: Flame,
        renderer: Renderer,
        color: torch.Tensor,
    ):
        out = flame.render(renderer=renderer, params=params)
        images = visualize_merged(
            s_color=color,
            t_color=out["color"],
            t_mask=out["mask"],
        )
        visualize_grid(images=images)
        img = wandb.Image(plt)
        plt.close()
        self.log_image(f"{self.mode}/{name}", [img])  # type: ignore

    # def init_logger(self, model: FLAME, cfg: DictConfig):
    #     config: dict = OmegaConf.to_container(cfg, resolve=True)  # type: ignore
    #     self.logger = wandb.init(
    #         dir=self.save_dir,
    #         project=self.project,
    #         entity=self.entity,
    #         group=self.group,
    #         tags=self.tags,
    #         name=self.name,
    #         config=config,
    #     )
    #     self.model = model

    def log(self, name: str, value: Any, step: None | int = None):
        self.log_metrics({name: value})

    def log_dict(self, value: dict, step: None | int = None):
        self.log_metrics(value)

    def log_path(self, folder: str, frame_idx: int, suffix: str):
        if self.capture_eval:
            return f"{self.mode}/{folder}/{frame_idx:05}.{suffix}"
        _f = f"{self.mode}/{folder}" if self.mode else folder
        return f"{_f}/{frame_idx:05}/{self.iter_step:05}.{suffix}"

    def iter_debug_idx(self, frame_idx: torch.Tensor):
        B = frame_idx.shape[0]
        for idx in range(B):
            yield B, idx, frame_idx[idx].item()

    def log_error(
        self,
        frame_idx: torch.Tensor,
        s_point: torch.Tensor,
        s_normal: torch.Tensor,
        t_mask: torch.Tensor,
        t_point: torch.Tensor,
        t_normal: torch.Tensor,
    ):
        """The max is an error of 10cm"""
        for _, b_idx, f_idx in self.iter_debug_idx(frame_idx):
            # point to point error map
            file_name = self.log_path("error_point_to_point", f_idx, "png")
            error_map = visualize_point2point_error(
                s_point=s_point[b_idx],
                t_point=t_point[b_idx],
                t_mask=t_mask[b_idx],
            )
            self.save_image(file_name, error_map)

            # point to plane error map
            file_name = self.log_path("error_point_to_plane", f_idx, "png")
            error_map = visualize_point2plane_error(
                s_point=s_point[b_idx],
                t_normal=t_normal[b_idx],
                t_point=t_point[b_idx],
                t_mask=t_mask[b_idx],
            )
            self.save_image(file_name, error_map)

            # normal distance
            file_name = self.log_path("error_normal", f_idx, "png")
            error_map = visualize_normal_error(
                s_normal=s_normal[b_idx],
                t_normal=t_normal[b_idx],
                t_mask=t_mask[b_idx],
            )
            self.save_image(file_name, error_map)

    def log_render(
        self,
        frame_idx: torch.Tensor,
        s_mask: torch.Tensor,
        s_point: torch.Tensor,
        s_color: torch.Tensor,
        t_mask: torch.Tensor,
        t_point: torch.Tensor,
        t_color: torch.Tensor,
        t_normal_image: torch.Tensor,
        t_depth_image: torch.Tensor,
    ):
        for _, b_idx, f_idx in self.iter_debug_idx(frame_idx):
            file_name = self.log_path("render_mask", f_idx, "png")
            self.save_image(file_name, t_mask[b_idx])

            file_name = self.log_path("render_depth", f_idx, "png")
            self.save_image(file_name, t_depth_image[b_idx])

            file_name = self.log_path("render_normal", f_idx, "png")
            self.save_image(file_name, t_normal_image[b_idx])

            file_name = self.log_path("render_color", f_idx, "png")
            self.save_image(file_name, t_color[b_idx])

            file_name = self.log_path("render_merged_depth", f_idx, "png")
            render_merged_depth = visualize_depth_merged(
                s_color=s_color,
                s_point=s_point,
                s_mask=s_mask,
                t_color=t_color,
                t_point=t_point,
                t_mask=t_mask,
            )
            self.save_image(file_name, render_merged_depth[b_idx])

            file_name = self.log_path("render_merged", f_idx, "png")
            render_merged = visualize_merged(
                s_color=s_color,
                t_color=t_color,
                t_mask=t_mask,
            )
            self.save_image(file_name, render_merged[b_idx])

    def log_input_batch(
        self,
        frame_idx: torch.Tensor,
        s_mask: torch.Tensor,
        s_point: torch.Tensor,
        s_color: torch.Tensor,
        s_normal: torch.Tensor,
    ):
        for _, b_idx, f_idx in self.iter_debug_idx(frame_idx):
            file_name = self.log_path("batch_mask", f_idx, "png")
            self.save_image(file_name, s_mask[b_idx])

            file_name = self.log_path("batch_depth", f_idx, "png")
            depth = Renderer.point_to_depth(s_point)
            depth_image = Renderer.depth_to_depth_image(depth)
            self.save_image(file_name, depth_image[b_idx])

            file_name = self.log_path("batch_normal", f_idx, "png")
            normal_image = Renderer.normal_to_normal_image(s_normal, s_mask)
            self.save_image(file_name, normal_image[b_idx])

            file_name = self.log_path("batch_color", f_idx, "png")
            self.save_image(file_name, s_color[b_idx])

    def log_metrics_stats(
        self,
        s_mask: torch.Tensor,
        s_point: torch.Tensor,
        t_mask: torch.Tensor,
        t_point: torch.Tensor,
        t_normal: torch.Tensor,
    ):
        point2plane = point2plane_distance(s_point, t_point, t_normal)
        self.log_metrics({f"{self.mode}/metric/point2plane": point2plane.mean()})

        # point2point = point2point_distance(m["point"], m["point_gt"])
        # self.log(f"{self.mode}/metric/point2point", point2point.mean())

        # lm_3d_dist = landmark_3d_distance(m["lm_3d_camera"], m["lm_3d_camera_gt"])
        # self.log(f"{self.mode}/metric/landmark_3d", lm_3d_dist.mean())

        # lm_2d_dist = landmark_3d_distance(m["lm_2d_ndc"], m["lm_2d_ndc_gt"])
        # self.log(f"{self.mode}/metric/landmark_2d", lm_2d_dist.mean())

        # # debug the regularization
        # params = ["expression_params", "shape_params"]
        # reg_dist = regularization_distance([m[p] for p in params])
        # self.log(f"{self.mode}/metric/regularization", reg_dist.mean())

        # # debug for each frame own loss curve
        # if self.mode == "sequential" and verbose:
        #     i = 0
        #     for _, b_idx, f_idx in self.iter_debug_idx(batch):
        #         pre = f"{self.mode}/metric/{f_idx:03}"
        #         j = mask[: b_idx + 1].sum()
        #         self.log(f"{pre}/point2plane", point2plane[i:j].mean())
        #         self.log(f"{pre}/point2point", point2point[i:j].mean())
        #         self.log(f"{pre}/landmark_3d", lm_3d_dist[b_idx].mean())
        #         self.log(f"{pre}/landmark_2d", lm_2d_dist[b_idx].mean())
        #         i = j

    # def log_gradients(self, verbose: bool = False):
    #     log = {}
    #     for p_name, weight in zip(self.optimizer._p_names, self.optimizer._params):
    #         grad = weight.grad

    #         log[f"{self.mode}/{p_name}/weight/mean"] = weight.mean()
    #         log[f"{self.mode}/{p_name}/weight/absmax"] = weight.abs().max()
    #         log[f"{self.mode}/{p_name}/grad/mean"] = grad.mean()
    #         log[f"{self.mode}/{p_name}/grad/absmax"] = grad.abs().max()

    #         if verbose:
    #             for i in range(weight.shape[-1]):
    #                 value = weight[:, i].mean()
    #                 log[f"{self.mode}/{p_name}/{i:03}/weight"] = value
    #             for i in range(grad.shape[-1]):
    #                 value = grad[:, i].mean()
    #                 log[f"{self.mode}/{p_name}/{i:03}/grad"] = value

    #     damping_factor = getattr(self.optimizer, "damping_factor", None)
    #     if damping_factor:
    #         log[f"{self.mode}/optimizer/damping_factor"] = damping_factor

    #     self.log_dict(log)

    def save_image(self, file_name: str, image: torch.Tensor):
        path = Path(self.save_dir) / file_name  # type: ignore
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image.detach().cpu().numpy()).save(path)

    def save_points(self, file_name: str, points: torch.Tensor):
        path = Path(self.save_dir) / file_name  # type: ignore
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            np.save(f, points.detach().cpu().numpy())

    def save_params(self, params: dict, frame_idx: torch.Tensor):
        assert len(frame_idx) == 1
        path = Path(self.save_dir) / f"params/{frame_idx.item():05}.pt"  # type: ignore
        path.parent.mkdir(parents=True, exist_ok=True)
        flame_params = {k: v.detach().cpu() for k, v in params.items()}
        torch.save(flame_params, path)

    def prepare_evaluation(
        self,
        renderer: Renderer,
        datamodule: DPHMDataModule,
        flame: Flame,
        params: dict,
        frame_idx: list[int],
    ):
        # full screen setup
        self.capture_eval = True
        renderer.update(scale=1)
        datamodule.update_dataset(
            camera=renderer.camera,
            rasterizer=renderer.rasterizer,
        )
        # dataset image
        datamodule.update_idxs(frame_idx)
        batch = datamodule.fetch()

        # render the full model
        self.mode = "eval"
        with torch.no_grad():
            m_out = flame(**params)
            out = flame.render(
                renderer=renderer,
                params=params,
            )

        self.log_error(
            frame_idx=batch["frame_idx"],
            s_point=batch["point"],
            s_normal=batch["normal"],
            t_mask=out["mask"],
            t_point=out["point"],
            t_normal=out["normal"],
        )
        self.log_render(
            frame_idx=batch["frame_idx"],
            s_mask=batch["mask"],
            s_point=batch["point"],
            s_color=batch["color"],
            t_mask=out["mask"],
            t_point=out["point"],
            t_color=out["color"],
            t_normal_image=out["normal_image"],
            t_depth_image=out["depth_image"],
        )
        self.log_input_batch(
            frame_idx=batch["frame_idx"],
            s_mask=batch["mask"],
            s_point=batch["point"],
            s_normal=batch["normal"],
            s_color=batch["color"],
        )
        self.save_params(
            params=params,
            frame_idx=batch["frame_idx"],
        )

    def log_tracking_video(self, name: str, framerate: int = 30, mode: str = "eval"):
        save_dir: str = self.save_dir  # type: ignore
        _video_dir = Path(save_dir) / mode / name
        assert _video_dir.exists()
        video_dir = str(_video_dir.resolve())
        video_path = str((Path(save_dir) / "video" / f"{name}.mp4").resolve())
        create_video(video_dir=video_dir, video_path=video_path, framerate=framerate)

    def log_tracker(self):
        statistics = self.time_tracker.compute_statistics()
        self.log_dict({f"{self.mode}/{k}": v["mean"] for k, v in statistics.items()})
