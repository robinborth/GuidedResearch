import torch

from lib.rasterizer import Fragments, Rasterizer
from lib.renderer.camera import Camera
from lib.trainer.timer import TimeTracker
from lib.utils.mesh import vertex_normals


class Renderer:
    def __init__(
        self,
        camera: Camera | None = None,
        rasterizer: Rasterizer | None = None,
        diffuse: list[float] = [0.5, 0.5, 0.5],
        specular: list[float] = [0.3, 0.3, 0.3],
        light: list[float] = [-1.0, 1.0, 0.0],
        color: int = 0,
        device: str = "cuda",
        **kwargs,
    ):
        """The rendering settings.

        The renderer is only initilized once, and updated for each rendering pass for
        the correct resolution camera. This is because creating openGL context is
        only done once.
        """
        self.camera = Camera() if camera is None else camera
        if rasterizer is None:
            rasterizer = Rasterizer(
                width=self.camera.width,
                height=self.camera.height,
            )
        self.rasterizer = rasterizer

        # rendering settings for shading
        self.diffuse = torch.tensor(diffuse, device=device)
        self.specular = torch.tensor(specular, device=device)
        self.light = torch.tensor(light, device=device)
        self.color = color

        self.time_tracker = TimeTracker()

    def to(self, device: str = "cpu"):
        self.camera = self.camera.to(device)
        self.diffuse = self.diffuse.to(device)
        self.specular = self.specular.to(device)
        self.light = self.light.to(device)
        return self

    def rasterize(self, vertices: torch.Tensor, faces: torch.Tensor) -> Fragments:
        homo_vertices = self.camera.convert_to_homo_coords(vertices)
        homo_clip_vertices = self.camera.clip_transform(homo_vertices)  # (B, V, 4)
        return self.rasterizer.rasterize(homo_clip_vertices, faces)

    def interpolate(
        self,
        vertices_idx: torch.Tensor,
        bary_coords: torch.Tensor,
        attributes: torch.Tensor,
    ):
        # access the vertex attributes
        B, H, W, _ = vertices_idx.shape  # (B, H, W, 3)
        _, V, D = attributes.shape  # (B, V, D)

        # Flatten the vertices_idx and bary_coords to reduce unnecessary operations
        flat_vertices_idx = vertices_idx.view(B, -1)  # (B, H*W*3)

        # Efficiently gather the vertex attributes in one step
        vertex_attributes = attributes.gather(
            1, flat_vertices_idx.unsqueeze(-1).expand(-1, -1, D)
        )  # (B, H*W*3, D)

        # Reshape gathered attributes to (B, H, W, 3, D) directly
        vertex_attributes = vertex_attributes.view(B, H, W, 3, D)

        # Perform the weighted sum using barycentric coordinates
        bary_coords = bary_coords.unsqueeze(-1)  # (B, H, W, 3, 1)
        attributes = (bary_coords * vertex_attributes).sum(dim=-2)  # (B, H, W, D)

        return attributes

    def mask_interpolate(
        self,
        vertices_idx: torch.Tensor,  # (B, H, W, 3)
        bary_coords: torch.Tensor,  # (B, H, W, 3)
        attributes: torch.Tensor,  # (B, V, D)
        mask: torch.Tensor,  # (B, H, W, 3)
    ):
        # access the vertex attributes
        B, V, D = attributes.shape  # (B, V, D)
        vertices_offset = V * torch.arange(B, device=vertices_idx.device)
        v_idx = vertices_idx.clone()
        v_idx += vertices_offset.view(B, 1, 1, 1)  # (B, H, W, 3)
        v_idx = v_idx[mask]  # (C, 3)
        vertex_attribute = attributes.reshape(-1, D)[v_idx]  # (C, 3, D)

        bary_coords = bary_coords[mask].unsqueeze(-1)  # (C, 3, 1)
        attributes = (bary_coords * vertex_attribute).sum(-2)  # (B, H, W, D)
        return attributes

    def render(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        attributes: torch.Tensor,
        fragments: Fragments | None = None,
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
        if fragments is None:
            fragments = self.rasterize(vertices, faces)
        attributes = self.interpolate(
            vertices_idx=fragments.vertices_idx,
            bary_coords=fragments.bary_coords,
            attributes=attributes,
        )
        return attributes, fragments.mask

    def render_point(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        fragments: Fragments | None = None,
    ):
        """Render the camera points map which camera z coordinate.

        Args:
            vertices (torch.Tensor): The vertices in camera coordinate system (B, V, 3)
            faces (torch.Tensor): The indexes of the vertices, e.g. the faces (F, 3)

        Returns:
            (torch.Tensor): Points in camera coordinate system where the depth is
                ranging from [0, inf] with dim (B, H, W, 3).
        """
        point, mask = self.render(vertices, faces, vertices, fragments)  # (B, H, W, 1)
        point[~mask, :] = 0
        return point, mask  # (B, H, W, 3),  (B, H, W)

    def render_depth(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        fragments: Fragments | None = None,
    ):
        """Render an depth map which camera z coordinate.

        Args:
            vertices (torch.Tensor): The vertices in camera coordinate system (B, V, 3)
            faces (torch.Tensor): The indexes of the vertices, e.g. the faces (F, 3)

        Returns:
            (torch.Tensor): Depth ranging from [0, inf] with dim (B, H, W).
        """
        point, mask = self.render_point(vertices, faces, fragments)
        depth = self.point_to_depth(point)
        return depth, mask

    def render_depth_image(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        fragments: Fragments | None = None,
    ):
        point, mask = self.render_point(vertices, faces, fragments)
        depth = self.point_to_depth(point)
        depth_image = self.depth_to_depth_image(depth)
        return depth_image, mask

    def render_normal(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        fragments: Fragments | None = None,
    ):
        """Render an depth map which camera z coordinate.

        Args:
            vertices (torch.Tensor): The vertices in camera coordinate system (B, V, 3)
            faces (torch.Tensor): The indexes of the vertices, e.g. the faces (F, 3)

        Returns:
            (torch.Tensor): Depth ranging from [0, inf] with dim (B, H, W, 3).
        """
        normals = vertex_normals(vertices, faces)
        normal, mask = self.render(
            vertices=vertices,
            faces=faces,
            attributes=normals,
            fragments=fragments,
        )  # (B, H', W', 3)
        return normal, mask

    def render_normal_image(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        fragments: Fragments | None = None,
    ):
        """Render normals in RGB space."""
        normal, mask = self.render_normal(vertices, faces, fragments)
        normal_image = self.normal_to_normal_image(normal, mask)
        return normal_image

    def render_color_image(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        fragments: Fragments | None = None,
    ):
        """Render the shaded image in RGB space."""
        normal, mask = self.render_normal(vertices, faces, fragments)
        color = self.normal_to_color_image(normal, mask)
        return color

    def render_full(self, vertices: torch.Tensor, faces: torch.Tensor):
        """Render all images."""
        # rasterize one time
        self.time_tracker.start("rasterize")
        fragments = self.rasterize(vertices.detach(), faces)
        # depth based
        self.time_tracker.start("depth_based", stop=True)
        point, _ = self.render_point(vertices, faces, fragments)
        depth = self.point_to_depth(point)
        depth_image = self.depth_to_depth_image(depth)
        # normal based
        self.time_tracker.start("normal_based", stop=True)
        normal, mask = self.render_normal(vertices, faces, fragments)
        normal_image = self.normal_to_normal_image(normal, mask)
        color_image = self.normal_to_color_image(normal, mask)
        self.time_tracker.stop()
        return {
            "mask": mask,
            "vertices_idx": fragments.vertices_idx,
            "bary_coords": fragments.bary_coords,
            "pix_to_face": fragments.pix_to_face,
            "point": point,
            "depth": depth,
            "normal": normal,
            "depth_image": depth_image,
            "normal_image": normal_image,
            "color": color_image,
        }

    ####################################################################################
    # Transformation Utils
    ####################################################################################

    @classmethod
    def point_to_depth(self, point):
        # we need to flip the z-axis because depth is positive
        return -point[:, :, :, 2]  # (H, W, 3)

    @classmethod
    def depth_to_depth_image(self, depth):
        return (depth.clip(0, 1) * 255).to(torch.uint8)

    @classmethod
    def normal_to_normal_image(self, normal, mask):
        normal_image = (((normal + 1) / 2) * 255).to(torch.uint8)
        normal_image[~mask] = 0
        return normal_image

    def normal_to_color_image(self, normal, mask):
        B, H, W, _ = normal.shape
        light = self.light / torch.norm(self.light)
        reflectance = (normal * light).sum(-1)[..., None]

        specular = reflectance * self.specular.expand(B, H, W, -1)
        diffuse = self.diffuse.expand(B, H, W, -1)

        color = specular + diffuse
        color = (torch.clip(color, 0, 1) * 255).to(torch.uint8)
        color[~mask] = 255

        return color
