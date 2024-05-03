import torch
import torch.nn as nn
from pytorch3d.renderer import MeshRasterizer, PerspectiveCameras, RasterizationSettings
from pytorch3d.structures import Meshes

from lib.renderer.camera import camera2normal


class Renderer(nn.Module):
    def __init__(
        self,
        K: torch.Tensor,
        image_scale: float = 1.0,
        image_width: int = 1920,
        image_height: int = 1080,
        diffuse: list[float] = [0.5, 0.5, 0.5],
        specular: list[float] = [0.3, 0.3, 0.3],
        light: list[float] = [-1.0, 1.0, 0.0],
        device: str = "cpu",
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

        self.H = image_width
        self.W = image_height
        self.device = device

        settings = RasterizationSettings(
            image_size=[self.H, self.W],
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=None,
            max_faces_per_bin=None,
            perspective_correct=False,
        )

        fx = K[0, 0] * image_scale
        fy = K[1, 1] * image_scale
        cx = K[0, 2] * image_scale
        cy = K[1, 2] * image_scale
        cameras = PerspectiveCameras(
            image_size=[[self.H, self.W]],
            focal_length=[(fx, fy)],
            principal_point=[(cx, cy)],
            in_ndc=False,
        )

        # define the rasterizer
        self.rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=settings,
        )

        # rendering settings for shading
        self.diffuse = torch.tensor(diffuse, device=device)
        self.specular = torch.tensor(specular, device=device)
        self.light = torch.tensor(light, device=device)

    def rasterize(self, vertices: torch.Tensor, faces: torch.Tensor):
        """Rendering of the attributes with mesh rasterization.

        NOTE: The rasterization only works on cpu currently.

        Args:
            faces (torch.Tensor): The indexes of the vertices, e.g. the faces (F, 3)
            vertices (torch.Tensor): The vertices in camera coordinate system (B, V, 3)

        Returns:
            (torch.Tensor, torch.Tensor) A tuple of pix_to_face cordinates of dim
            (B, H, W) and the coresponding bary coordinates of dim (B, H, W, 3).
        """
        verts = vertices.cpu().clone()
        verts[:, :1] = -verts[:, :1]
        faces = faces.expand(verts.shape[0], -1, -1).cpu()
        meshes = Meshes(verts=verts, faces=faces)
        fragments = self.rasterizer(meshes)
        # opengl convension is that we store the down row first
        pix_to_face = torch.flip(fragments.pix_to_face.squeeze(-1), [1])
        bary_coords = torch.flip(fragments.bary_coords.squeeze(-2), [1])
        return pix_to_face.to(self.device), bary_coords.to(self.device)

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
        pix2face, b_coords = self.rasterize(vertices, faces)
        vertices_idx = faces[pix2face]  # (B, H, W, 3)

        # access the vertex attributes
        B, H, W, C = vertices_idx.shape  # (B, H, W, 3)
        _, _, D = attributes.shape  # (B, V, D)
        v_idx = vertices_idx.reshape(B, -1)  # (B, *)
        b_idx = torch.arange(v_idx.size(0)).unsqueeze(1).to(v_idx)
        vertex_attribute = attributes[b_idx, v_idx]  # (B, *, D)
        vertex_attribute = vertex_attribute.reshape(B, H, W, C, D)  # (B, H, W, 3, D)

        bary_coords = b_coords.unsqueeze(-1)  # (B, H, W, 3, 1)
        attributes = (bary_coords * vertex_attribute).sum(-2)  # (B, H, W, D)

        # forground mask
        mask = pix2face != -1

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

    def depth_to_depth_image(self, depth):
        return (depth.clip(0, 1) * 255).to(torch.uint8)

    def render_depth_image(self, point):
        depth, mask = self.point_to_depth(point)
        depth_image = self.depth_to_depth_image(depth)
        return depth_image, mask

    def point_to_normal(self, point):
        normal, normal_mask = camera2normal(point)
        return normal, normal_mask

    def render_normal(self, vertices: torch.Tensor, faces: torch.Tensor):
        """Render an depth map which camera z coordinate.

        Args:
            vertices (torch.Tensor): The vertices in camera coordinate system (B, V, 3)
            faces (torch.Tensor): The indexes of the vertices, e.g. the faces (F, 3)

        Returns:
            (torch.Tensor): Depth ranging from [0, inf] with dim (B, H, W, 3).
        """
        point, _ = self.render_point(vertices=vertices, faces=faces)
        normal, normal_mask = self.point_to_normal(point)  # (B, H', W', 3)
        return normal, normal_mask

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
        shading, mask = self.normal_to_shading(normal, mask)
        return shading, mask

    def render_full(self, vertices: torch.Tensor, faces: torch.Tensor):
        """Render all images."""
        point, mask = self.render_point(vertices, faces)
        depth = self.point_to_depth(point)
        depth_image = self.depth_to_depth_image(depth)
        normal, normal_mask = self.point_to_normal(point)
        normal_image = self.normal_to_normal_image(normal, normal_mask)
        shading_image = self.normal_to_shading_image(normal, normal_mask)
        return {
            "mask": mask,
            "point": point,
            "depth": depth,
            "depth_image": depth_image,
            "normal": normal,
            "normal_mask": normal_mask,
            "normal_image": normal_image,
            "shading_image": shading_image,
        }
