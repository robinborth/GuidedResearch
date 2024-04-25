import torch
import torch.nn as nn
from pytorch3d.renderer import MeshRasterizer, PerspectiveCameras, RasterizationSettings
from pytorch3d.structures import Meshes

from lib.renderer.camera import load_intrinsics


class Rasterizer(nn.Module):
    def __init__(
        self,
        data_dir: str = "data/",
        scale_factor: int = 1,
        image_width: int = 1920,
        image_height: int = 1080,
    ):
        super().__init__()
        # specify the output dimension of the image
        self.image_size = (image_height // scale_factor, image_width // scale_factor)
        settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=None,
            max_faces_per_bin=None,
            perspective_correct=False,
        )

        # specify the camera intrinsic from camera to pixel
        K = load_intrinsics(data_dir=data_dir)
        cameras = PerspectiveCameras(
            image_size=[(image_height, image_width)],
            focal_length=[(K["fx"], K["fy"])],
            principal_point=[(K["cx"], K["cy"])],
            in_ndc=False,
        )

        # define the rasterizer
        self._rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=settings,
        )

    def forward(self, vertices: torch.Tensor, faces: torch.Tensor):
        verts = vertices.clone()
        verts[:, :1] = -verts[:, :1]
        meshes = Meshes(verts=[verts], faces=[faces])
        fragments = self._rasterizer(meshes)
        # up/down problem https://github.com/facebookresearch/pytorch3d/issues/78
        pix_to_face = torch.flip(fragments.pix_to_face.squeeze(), [0])
        bary_coords = torch.flip(fragments.bary_coords.squeeze(), [0])
        return pix_to_face, bary_coords
