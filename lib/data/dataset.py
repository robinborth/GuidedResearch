import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from lib.model.flame import FLAME
from lib.rasterizer import Rasterizer
from lib.renderer.camera import Camera
from lib.renderer.renderer import Renderer
from lib.utils.loader import (
    load_color,
    load_depth,
    load_mediapipe_landmark_2d,
    load_mediapipe_landmark_3d,
    load_normal,
)


class DPHMDataset(Dataset):
    def __init__(
        self,
        # rasterizer settings
        camera: Camera,
        # dataset settings
        data_dir: str = "/data",
        optimize_frames: int = 1,
        start_frame_idx: int = 0,
        **kwargs,
    ):
        self.optimize_frames = optimize_frames
        self.start_frame_idx = start_frame_idx
        self.camera = camera
        self.data_dir = data_dir

    def iter_frame_idx(self):
        yield from range(
            self.start_frame_idx, self.start_frame_idx + self.optimize_frames
        )

    def load_lm_2d_ndc(self):
        self.lm_2d_ndc = []
        for frame_idx in self.iter_frame_idx():
            landmarks = load_mediapipe_landmark_2d(self.data_dir, idx=frame_idx)
            # the landmarks needs to be in ndc space hence bottom left is (-1,-1)
            landmarks[:, 0] = (landmarks[:, 0] * 2) - 1
            landmarks[:, 1] = -((landmarks[:, 1] * 2) - 1)
            # landmarks[:, 1] = ((landmarks[:, 1] - 1) * 2) + 1
            self.lm_2d_ndc.append(landmarks)

    def load_lm_3d_camera(self):
        self.lm_3d_camera = []
        for frame_idx in self.iter_frame_idx():
            landmarks = load_mediapipe_landmark_3d(self.data_dir, idx=frame_idx)
            landmarks[:, 1] = -landmarks[:, 1]
            landmarks[:, 2] = -landmarks[:, 2]
            self.lm_3d_camera.append(landmarks)

    def load_color(self):
        self.color = []
        for frame_idx in self.iter_frame_idx():
            image = load_color(
                data_dir=self.data_dir,
                idx=frame_idx,
                return_tensor="pt",
            )  # (H, W, 3)
            image = v2.functional.resize(
                inpt=image.permute(2, 0, 1),
                size=(self.camera.height, self.camera.width),
            ).permute(1, 2, 0)
            self.color.append(image.to(torch.uint8))  # (H',W',3)

    def load_point(self):
        self.point = []
        self.mask = []
        for frame_idx in self.iter_frame_idx():
            depth = load_depth(
                data_dir=self.data_dir,
                idx=frame_idx,
                return_tensor="pt",
                smooth=True,
            )
            point, mask = self.camera.depth_map_transform(depth)
            self.point.append(point)
            self.mask.append(mask)

    def load_normal(self):
        self.normal = []
        for frame_idx in self.iter_frame_idx():
            normal = load_normal(
                data_dir=self.data_dir,
                idx=frame_idx,
                return_tensor="pt",
                smooth=True,
            )
            normal = v2.functional.resize(
                inpt=normal.permute(2, 0, 1),
                size=(self.camera.height, self.camera.width),
            ).permute(1, 2, 0)
            # to make them right-hand and follow camera space convention +X right +Y up
            # +Z towards the camera
            normal[:, :, 1] = -normal[:, :, 1]
            normal[:, :, 2] = -normal[:, :, 2]
            self.normal.append(normal)

    def __len__(self) -> int:
        return self.optimize_frames

    def __getitem__(self, idx: int):
        raise NotImplementedError()


class DPHMPointDataset(DPHMDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.load_point()
        self.load_normal()
        self.load_lm_2d_ndc()
        self.load_lm_3d_camera()
        self.load_color()

    def __getitem__(self, idx: int):
        # (H', W', 3) this is scaled
        mask = self.mask[idx]
        point = self.point[idx]
        normal = self.normal[idx]
        color = self.color[idx]
        lm_2d_ndc = self.lm_2d_ndc[idx]
        lm_3d_camera = self.lm_3d_camera[idx]
        return {
            "shape_idx": 0,
            "frame_idx": idx,
            "mask": mask,
            "point": point,
            "normal": normal,
            "color": color,
            "lm_2d_ndc": lm_2d_ndc,
            "lm_3d_camera": lm_3d_camera,
        }


class FLAMEDataset(DPHMDataset):
    def __init__(
        self,
        rasterizer: Rasterizer,
        flame_dir: str = "/flame",
        num_shape_params: int = 100,
        num_expression_params: int = 50,
        optimize_frames: int = 1,
        optimize_shapes: int = 1,
        vertices_mask: str = "face",
        **kwargs,
    ):
        super().__init__(**kwargs, optimize_frames=optimize_frames)

        self.shape_idx = 0
        self.frame_idx = 0

        flame = FLAME(
            flame_dir=flame_dir,
            num_shape_params=num_shape_params,
            num_expression_params=num_expression_params,
            optimize_frames=optimize_frames,
            optimize_shapes=optimize_shapes,
            vertices_mask=vertices_mask,
        ).to("cuda")
        flame.init_params_flame(0.0)
        flame.init_renderer(
            camera=self.camera,
            rasterizer=rasterizer,
            diffuse=[0.6, 0.0, 0.0],
            specular=[0.5, 0.0, 0.0],
        )

        model = flame.model_step(frame_idx=self.frame_idx, shape_idx=self.shape_idx)
        vertices = model["vertices"]
        lm_2d_ndc = model["lm_2d_ndc"]
        lm_3d_camera = model["lm_3d_camera"]

        render = flame.renderer.render_full(vertices, flame.masked_faces(vertices))
        self.mask = render["mask"][0].detach().cpu().numpy()
        self.point = render["point"][0].detach().cpu().numpy()
        self.normal = render["normal"][0].detach().cpu().numpy()
        self.color = render["color"][0].detach().cpu().numpy()
        self.color[~self.mask, :] = 255
        self.lm_3d_camera = lm_3d_camera[0].detach().cpu().numpy()
        self.lm_2d_ndc = lm_2d_ndc[0].detach().cpu().numpy()

        self.shape_params = flame.shape_params.weight[self.shape_idx]
        self.expression_params = flame.expression_params.weight[self.frame_idx]
        self.global_pose = flame.global_pose.weight[self.frame_idx]
        self.neck_pose = flame.neck_pose.weight[self.frame_idx]
        self.jaw_pose = flame.jaw_pose.weight[self.frame_idx]
        self.eye_pose = flame.eye_pose.weight[self.frame_idx]
        self.transl = flame.transl.weight[self.frame_idx]
        self.scale = flame.scale.weight[self.frame_idx]

    def __getitem__(self, idx: int):
        return {
            # data params
            "shape_idx": self.shape_idx,
            "frame_idx": self.frame_idx,
            "mask": self.mask,
            "point": self.point,
            "normal": self.normal,
            "color": self.color,
            "lm_2d_ndc": self.lm_2d_ndc,
            "lm_3d_camera": self.lm_3d_camera,
            # flame params
            "shape_params": self.shape_params,
            "expression_params": self.expression_params,
            "global_pose": self.global_pose,
            "neck_pose": self.neck_pose,
            "jaw_pose": self.jaw_pose,
            "eye_pose": self.eye_pose,
            "transl": self.transl,
            "scale": self.scale,
        }
