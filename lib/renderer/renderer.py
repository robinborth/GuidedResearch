import torch
import torch.nn as nn

from lib.rasterizer import Rasterizer
from lib.utils.mesh import weighted_vertex_normals


class Renderer(nn.Module):
    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        diffuse: list[float] = [0.5, 0.5, 0.5],
        specular: list[float] = [0.3, 0.3, 0.3],
        light: list[float] = [-1.0, 1.0, 0.0],
        device: str = "cpu",
        **kwargs,
    ):
        """The rendering settings.

        Args:
            K (dict): NOTE: The scaled intrinsic based on H', W' of dim (3, 3).
            image_width (int, optional): _description_. Defaults to 1920.
            image_height (int, optional): _description_. Defaults to 1080.
            diffuse (list[float]): _description_. Defaults to [0.5, 0.5, 0.5].
            specular (list[float]): _description_. Defaults to [0.3, 0.3, 0.3].
            light (list[float]): _description_. Defaults to [-1.0, 1.0, 0.0].
            device (str): _description_. Defaults to "cpu".
        """
        super().__init__()

        # create the rasterizer
        self.rasterizer = Rasterizer(width=width, height=height)

        # rendering settings for shading
        self.diffuse = torch.tensor(diffuse, device=device)
        self.specular = torch.tensor(specular, device=device)
        self.light = torch.tensor(light, device=device)

    def render(
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
        # (B, H, W), (B, H, W, 3)
        fragments = self.rasterizer.rasterize(vertices, faces)
        vertices_idx = faces[fragments.pix_to_face]  # (B, H, W, 3)

        # access the vertex attributes
        B, H, W, _ = vertices_idx.shape  # (B, H, W, 3)
        _, _, D = attributes.shape  # (B, V, D)
        v_idx = vertices_idx.reshape(B, -1)  # (B, *)
        b_idx = torch.arange(v_idx.size(0)).unsqueeze(1).to(v_idx)
        vertex_attribute = attributes[b_idx, v_idx]  # (B, *, D)
        vertex_attribute = vertex_attribute.reshape(B, H, W, 3, D)  # (B, H, W, 3, D)

        bary_coords = fragments.bary_coords.unsqueeze(-1)  # (B, H, W, 3, 1)
        attributes = (bary_coords * vertex_attribute).sum(-2)  # (B, H, W, D)

        # forground mask
        mask = fragments.pix_to_face != -1

        return attributes, mask

    def render_point(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
    ):
        """Render the camera points map which camera z coordinate.

        Args:
            vertices (torch.Tensor): The vertices in camera coordinate system (B, V, 3)
            faces (torch.Tensor): The indexes of the vertices, e.g. the faces (F, 3)

        Returns:
            (torch.Tensor): Points in camera coordinate system where the depth is
                ranging from [0, inf] with dim (B, H, W, 3).
        """
        point, mask = self.render(vertices, faces, vertices)  # (B, H, W, 1), (B, H, W)
        point[~mask, :] = 0
        return point, mask  # (B, H, W, 3),  (B, H, W)

    @classmethod
    def point_to_depth(self, point):
        return point[:, :, :, 2]  # (H, W, 3)

    def render_depth(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
    ):
        """Render an depth map which camera z coordinate.

        Args:
            vertices (torch.Tensor): The vertices in camera coordinate system (B, V, 3)
            faces (torch.Tensor): The indexes of the vertices, e.g. the faces (F, 3)

        Returns:
            (torch.Tensor): Depth ranging from [0, inf] with dim (B, H, W).
        """
        point, mask = self.render_point(vertices, faces)
        depth = self.point_to_depth(point)
        return depth, mask

    @classmethod
    def depth_to_depth_image(self, depth):
        return (depth.clip(0, 1) * 255).to(torch.uint8)

    def render_depth_image(self, point):
        depth, mask = self.point_to_depth(point)
        depth_image = self.depth_to_depth_image(depth)
        return depth_image, mask

    def render_normal(self, vertices: torch.Tensor, faces: torch.Tensor):
        """Render an depth map which camera z coordinate.

        Args:
            vertices (torch.Tensor): The vertices in camera coordinate system (B, V, 3)
            faces (torch.Tensor): The indexes of the vertices, e.g. the faces (F, 3)

        Returns:
            (torch.Tensor): Depth ranging from [0, inf] with dim (B, H, W, 3).
        """
        vertex_normals = weighted_vertex_normals(vertices, faces)
        normal, mask = self.render(
            vertices=vertices,
            faces=faces,
            attributes=vertex_normals,
        )  # (B, H', W', 3)
        return normal, mask

    @classmethod
    def normal_to_normal_image(self, normal, mask):
        normal_image = (((normal + 1) / 2) * 255).to(torch.uint8)
        normal_image[~mask] = 0
        return normal_image

    def render_normal_image(self, vertices: torch.Tensor, faces: torch.Tensor):
        """Render normals in RGB space."""
        normal, mask = self.render_normal(vertices, faces)
        normal_image = self.normal_to_normal_image(normal, mask)
        return normal_image

    def normal_to_shading_image(self, normal, mask):
        B, H, W, _ = normal.shape
        light = self.light / torch.norm(self.light)
        reflectance = (normal * light).sum(-1)[..., None]

        specular = reflectance * self.specular.expand(B, H, W, -1)
        diffuse = self.diffuse.expand(B, H, W, -1)

        shading = specular + diffuse
        shading = (torch.clip(shading, 0, 1) * 255).to(torch.uint8)
        shading[~mask] = 0

        return shading

    def render_shading_image(self, vertices: torch.Tensor, faces: torch.Tensor):
        """Render the shaded image in RGB space."""
        normal, mask = self.render_normal(vertices, faces)
        shading = self.normal_to_shading_image(normal, mask)
        return shading

    def render_full(self, vertices: torch.Tensor, faces: torch.Tensor):
        """Render all images."""
        # depth based
        point, _ = self.render_point(vertices, faces)
        depth = self.point_to_depth(point)
        depth_image = self.depth_to_depth_image(depth)
        # normal based
        normal, mask = self.render_normal(vertices, faces)
        normal_image = self.normal_to_normal_image(normal, mask)
        shading_image = self.normal_to_shading_image(normal, mask)
        return {
            "mask": mask,
            "point": point,
            "depth": depth,
            "normal": normal,
            "depth_image": depth_image,
            "normal_image": normal_image,
            "shading_image": shading_image,
        }
