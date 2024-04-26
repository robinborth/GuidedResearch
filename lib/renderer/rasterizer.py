import lightning as L
import torch
import torch.nn as nn
from pytorch3d.renderer import MeshRasterizer, PerspectiveCameras, RasterizationSettings
from pytorch3d.structures import Meshes

from lib.renderer.camera import load_intrinsics


class Rasterizer(L.LightningModule):
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
        """Rendering of the attributes with mesh rasterization.

        Args:
            faces (torch.Tensor): The indexes of the vertices, e.g. the faces (F, 3)
            vertices (torch.Tensor): The vertices in camera coordinate system (B, V, 3)

        Returns:
            (torch.Tensor, torch.Tensor) A tuple of pix_to_face cordinates of dim
            (B, H, W) and the coresponding bary coordinates of dim (B, H, W, 3).
        """
        verts = vertices.clone()
        verts[:, :1] = -verts[:, :1]
        meshes = Meshes(verts=verts, faces=faces.expand(verts.shape[0], -1, -1))
        fragments = self._rasterizer(meshes)
        # opengl convension is that we store the down row first
        pix_to_face = torch.flip(fragments.pix_to_face.squeeze(-1), [1])
        bary_coords = torch.flip(fragments.bary_coords.squeeze(-2), [1])
        return pix_to_face, bary_coords
