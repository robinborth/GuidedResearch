import torch
import torch.nn as nn
import trimesh
from renderer.rasterizer import MeshRasterizer
from torchvision.transforms import v2


class Renderer(nn.Module):
    def __init__(
        self,
        rasterizer: MeshRasterizer,
        diffuse: list[float] | None = None,
        specular: list[float] | None = None,
        light: list[float] | None = None,
    ):
        super().__init__()
        self.rasterizer = rasterizer
        diffuse = diffuse if diffuse is not None else [0.5, 0.5, 0.5]
        self.diffuse = torch.tensor(diffuse)
        specular = specular if specular is not None else [0.3, 0.3, 0.3]
        self.specular = torch.tensor(specular)
        light = light if light is not None else [1.0, 1.0, 0.0]  # world coord space
        self.light = torch.tensor(light)

    def forward(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        attributes: torch.Tensor,
    ):
        """Rendering of the attributes with mesh rasterization.

        Args:
            vertices (torch.Tensor): The vertices in camera coordinate system (V, 3)
            faces (torch.Tensor): The indexes of the vertices, e.g. the faces (F, 3)
            attributes (torch.Tensor): The attributes per vertex of dim (V, D)

        Returns:
            (torch.Tensor): The output is the image plane filled with the attributes,
                that are barycentric interpolated, hence the dim is (H, W, D). Not that
                we have a row-major matrix representation.
        """
        pix_to_face, bary_coords = self.rasterizer(vertices, faces)  # (H, W), (H, W, 3)
        vertices_idx = faces[pix_to_face]  # (H, W, 3)
        vertex_attribute = attributes[vertices_idx]  # (H, W, 3, D)
        attribute_map = (bary_coords.unsqueeze(-1) * vertex_attribute).sum(-2)
        return attribute_map, pix_to_face != -1

    def render_depth(self, vertices: torch.Tensor, faces: torch.Tensor):
        """Render an depth map which camera z coordinate.

        Args:
            vertices (torch.Tensor): The vertices in camera coordinate system (V, 3)
            faces (torch.Tensor): The indexes of the vertices, e.g. the faces (F, 3)

        Returns:
            (torch.Tensor): Depth ranging from [0, inf] with dim (H, W).
        """
        attributes = -vertices[:, 2].unsqueeze(-1)  # depth values of dim (V, 1)
        depth, mask = self.forward(vertices, faces, attributes)  # (H, W, 1), (H, W)
        depth[~mask, :] = 0  # TODO is this a good default value?
        return depth.squeeze(-1)  # (H, W)

    def render_normal(self, vertices: torch.Tensor, faces: torch.Tensor):
        """Render an normalized normal map.

        Args:
            vertices (torch.Tensor): The vertices in camera coordinate system (V, 3)
            faces (torch.Tensor): The indexes of the vertices, e.g. the faces (F, 3)

        Returns:
            (torch.Tensor): Normals ranging from [-1, 1] and have unit lenght, the dim
                is of the canvas hence (H, W).
        """
        tmesh = trimesh.Trimesh(vertices.detach().cpu().numpy(), faces)
        attributes = torch.tensor(tmesh.vertex_normals, dtype=vertices.dtype)
        normals, mask = self.forward(vertices, faces, attributes)  # (H, W, 3)
        normals = normals / torch.norm(normals, dim=-1)[..., None]
        normals[~mask, :] = 0
        return normals

    def render_normal_image(self, vertices: torch.Tensor, faces: torch.Tensor):
        """Render normals in RGB space."""
        normals = self.render_normal(vertices, faces)
        background_mask = normals.sum(-1) == 0
        normals = (((normals + 1) / 2) * 255).to(torch.uint8)
        normals[background_mask] = 0
        return normals

    def render_shader_image(self, vertices: torch.Tensor, faces: torch.Tensor):
        """Render the shaded image in RGB space."""
        normals = self.render_normal(vertices, faces)
        background_mask = normals.sum(-1) == 0
        H, W, _ = normals.shape

        light = self.light / torch.norm(self.light)
        reflectance = (normals * light).sum(-1)
        specular = reflectance[..., None] * self.specular.expand(H, W, -1)
        diffuse = self.diffuse.expand(H, W, -1)

        image = specular + diffuse
        image = (torch.clip(image, 0, 1) * 255).to(torch.uint8)
        image[background_mask, :] = 0

        return image

    def render_color_image(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        image: torch.tensor,
    ):
        """Render the flame model on top of the resized color image in RGB space.

        Args:
            vertices (torch.Tensor): The vertices in camera coordinate system (V, 3)
            faces (torch.Tensor): The indexes of the vertices, e.g. the faces (F, 3)
            image (torch.tensor): The color image of original dim (H', W', 3).
        """
        flame_image = self.render_shader_image(vertices, faces)
        flame_mask = flame_image.sum(-1) != 0
        background = v2.functional.resize(
            inpt=image.permute(2, 0, 1),
            size=self.rasterizer.image_size,
        ).permute(1, 2, 0)
        background[flame_mask] = flame_image[flame_mask]
        return background
