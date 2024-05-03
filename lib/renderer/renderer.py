import lightning as L
import torch
import torch.nn as nn
from torchvision.transforms import v2

from lib.model.utils import bary_coord_interpolation, flame_faces_mask
from lib.renderer.camera import camera2normal
from lib.renderer.rasterizer import Rasterizer


class Renderer(L.LightningModule):
    def __init__(
        self,
        rasterizer: Rasterizer | None = None,
        diffuse: list[float] | None = None,
        specular: list[float] | None = None,
        light: list[float] | None = None,
    ):
        super().__init__()
        self.rasterizer = rasterizer if rasterizer is not None else Rasterizer()
        diffuse = diffuse if diffuse is not None else [0.5, 0.5, 0.5]
        self.diffuse = torch.tensor(diffuse)
        specular = specular if specular is not None else [0.3, 0.3, 0.3]
        self.specular = torch.tensor(specular)
        light = light if light is not None else [-1.0, 1.0, 0.0]  # world coord space
        self.light = torch.tensor(light)

    def forward(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        attributes: torch.Tensor,
    ):
        """Rendering of the attributes with mesh rasterization.

        Args:
            faces (torch.Tensor): The indexes of the vertices, e.g. the faces (F, 3)
            vertices (torch.Tensor): The vertices in camera coordinate system (B, V, 3)
            attributes (torch.Tensor): The attributes per vertex of dim (B, V, D)

        Returns:
            (torch.Tensor): The output is the image plane filled with the attributes,
                that are barycentric interpolated, hence the dim is (H, W, D). Not that
                we have a row-major matrix representation.
        """
        pix2face, b_coords = self.rasterizer(
            vertices.cpu(), faces.cpu()
        )  # (B, H, W), (B, H, W, 3)
        vertices_idx = faces[pix2face]  # (B, H, W, 3)

        # access the vertex attributes
        B, H, W, C = vertices_idx.shape  # (B, H, W, 3)
        _, _, D = attributes.shape  # (B, V, D)
        v_idx = vertices_idx.reshape(B, -1)  # (B, *)
        b_idx = torch.arange(v_idx.size(0)).unsqueeze(1).to(v_idx)
        vertex_attribute = attributes[b_idx, v_idx]  # (B, *, D)
        vertex_attribute = vertex_attribute.reshape(B, H, W, C, D)  # (B, H, W, 3, D)

        bary_coords = b_coords.unsqueeze(-1).to(self.device)  # (B, H, W, 3, 1)
        attributes = (bary_coords * vertex_attribute).sum(-2)  # (B, H, W, D)

        return attributes, pix2face != -1

    def render_depth(self, vertices: torch.Tensor, faces: torch.Tensor):
        """Render an depth map which camera z coordinate.

        Args:
            vertices (torch.Tensor): The vertices in camera coordinate system (B, V, 3)
            faces (torch.Tensor): The indexes of the vertices, e.g. the faces (F, 3)

        Returns:
            (torch.Tensor): Depth ranging from [0, inf] with dim (B, H, W).
        """
        attributes = vertices[:, :, 2].unsqueeze(-1)  # depth values of dim (B, V, 1)
        depth, mask = self.forward(vertices, faces, attributes)  # (H, W, 1), (H, W)
        depth[~mask, :] = 0  # TODO is this a good default value?
        return depth.squeeze(-1)  # (H, W)

    def render_point_image(self, vertices: torch.Tensor, faces: torch.Tensor):
        """Render an depth map which camera z coordinate.

        Args:
            vertices (torch.Tensor): The vertices in camera coordinate system (B, V, 3)
            faces (torch.Tensor): The indexes of the vertices, e.g. the faces (F, 3)

        Returns:
            (torch.Tensor): Depth ranging from [0, inf] with dim (B, H, W, 3).
        """
        point, mask = self.forward(vertices, faces, vertices)  # (B, H, W, 1), (B, H, W)
        point[~mask, :] = 0
        return point  # (B, H, W, 3)

    def render_normal(self, vertices: torch.Tensor, faces: torch.Tensor):
        """Render an depth map which camera z coordinate.

        Args:
            vertices (torch.Tensor): The vertices in camera coordinate system (B, V, 3)
            faces (torch.Tensor): The indexes of the vertices, e.g. the faces (F, 3)

        Returns:
            (torch.Tensor): Depth ranging from [0, inf] with dim (B, H, W, 3).
        """
        point = self.render_point_image(vertices=vertices, faces=faces)
        return camera2normal(point)  # (B, H', W', 3)

    # def render_normal(
    #     self,
    #     vertices: torch.Tensor,
    #     faces: torch.Tensor,
    #     vertices_mask: torch.Tensor | None = None,
    # ):
    #     """Render an normalized normal map.

    #     Args:
    #         vertices (torch.Tensor): The vertices in camera coordinate system (V, 3)
    #         faces (torch.Tensor): The indexes of the vertices, e.g. the faces (F, 3)
    #         vertices_mask (torch.Tensor): The idx of the vertices that should be
    #             included in the rendering and computation.

    #     Returns:
    #         (torch.Tensor): Normals ranging from [-1, 1] and have unit lenght, the dim
    #             is of the canvas hence (H, W).
    #     """

    #     # tmesh = trimesh.Trimesh(
    #     #     vertices=vertices.detach().cpu().numpy(),
    #     #     faces=faces.detach().cpu().numpy(),
    #     # )
    #     tmesh = ...
    #     attributes = torch.tensor(tmesh.vertex_normals, dtype=vertices.dtype)
    #     faces_mask = flame_faces_mask(vertices, faces, vertices_mask)
    #     normals, mask = self.forward(vertices, faces[faces_mask], attributes)  # (H,W,3)
    #     normals = normals / torch.norm(normals, dim=-1)[..., None]
    #     normals[~mask, :] = 0
    #     return normals

    # def render_normal_image(
    #     self,
    #     vertices: torch.Tensor,
    #     faces: torch.Tensor,
    #     vertices_mask: torch.Tensor | None = None,
    # ):
    #     """Render normals in RGB space."""
    #     normals = self.render_normal(vertices, faces, vertices_mask)
    #     background_mask = normals.sum(-1) == 0
    #     normals = (((normals + 1) / 2) * 255).to(torch.uint8)
    #     normals[background_mask] = 0
    #     return normals

    # def render_shader_image(
    #     self,
    #     vertices: torch.Tensor,
    #     faces: torch.Tensor,
    #     vertices_mask: torch.Tensor | None = None,
    # ):
    #     """Render the shaded image in RGB space."""
    #     normals = self.render_normal(vertices, faces, vertices_mask)
    #     background_mask = normals.sum(-1) == 0
    #     H, W, _ = normals.shape

    #     light = self.light / torch.norm(self.light)
    #     reflectance = (normals * light).sum(-1)
    #     specular = reflectance[..., None] * self.specular.expand(H, W, -1)
    #     diffuse = self.diffuse.expand(H, W, -1)

    #     image = specular + diffuse
    #     image = (torch.clip(image, 0, 1) * 255).to(torch.uint8)
    #     image[background_mask, :] = 0

    #     return image

    def render_color_image(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        image: torch.Tensor,
        vertices_mask: torch.Tensor | None = None,
        rescale: bool = False,
    ):
        """Render the flame model on top of the resized color image in RGB space.

        Args:
            vertices (torch.Tensor): The vertices in camera coordinate system (V, 3)
            faces (torch.Tensor): The indexes of the vertices, e.g. the faces (F, 3)
            image (torch.tensor): The color image of original dim (H', W', 3).
        """
        flame_image = self.render_shader_image(vertices, faces, vertices_mask)
        flame_mask = flame_image.sum(-1) != 0
        if rescale:
            image = v2.functional.resize(
                inpt=image.permute(2, 0, 1),
                size=self.rasterizer.image_size,
            )
            image = image.permute(1, 2, 0).to(torch.uint8)
        image[flame_mask] = flame_image[flame_mask]
        return image
