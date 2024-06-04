import math

import torch
from torchvision.transforms import v2


class Camera:
    """
    For the camera there are four different coordinate systems (or spaces):
    - Camera space: We follow OpenGL coordinate system with the
        right-hand rule where we have +X points right, +Y points up and +Z
        point to the camera.
    - Cilp space: Any coordinate outside within the range is being clipped.
        We use this coordinate system as an input for the rasterizer. Note
        That we use homogeneous coordinates, hence the w value (x,y,z,w) is
        used for clipping -w < x,y,z < w. Note that we convert here from
        right-hand rule to left-hand rule, because +Z now points awy from
        the camera.
    - NDC space: This is the normalzied Clip space where each coordiante
        in the frustrum is between (-1,1). We don't performe the
        transformation ourselv we let OpenGL do that.
    - Screen space: The representation of the frustrum defined in pixel space
        instead of a normalzied space, where the top left pixel is (0,0) and
        the bottom right corner is (W,H).

    Args:
        K: Note that the intrinsics matrix needs to be the default matrix, the
            scaling happens inside the camera model.
    """

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        scale: float = 1.0,
        fov_y: float = 45.0,
        near: float = 0.01,
        far: float = 100.0,
        K: torch.Tensor | None = None,
        device: str = "cuda",
    ):
        self.original_width = width
        self.original_height = height
        self.scale = scale
        self.width = int(self.original_width * self.scale)
        self.height = int(self.original_height * self.scale)
        self.near = near
        self.far = far
        self.fov_y = fov_y
        self.K = K
        self.device = device
        self.set_perspective_projection()

    def update(self, scale: float = 1.0):
        self.scale = scale
        self.width = int(self.original_width * self.scale)
        self.height = int(self.original_height * self.scale)
        self.set_perspective_projection()

    def set_perspective_projection(self):
        if self.K is None:
            self.projection_matrix = self.fov_perspective_projection(
                fov_y=self.fov_y,
                width=self.width,
                height=self.height,
                near=self.near,
                far=self.far,
            )
        else:
            self.projection_matrix = self.intrinsics_perspective_projection(
                K=self.K,
                scale=self.scale,
                width=self.width,
                height=self.height,
                near=self.near,
                far=self.far,
            )

    def fov_perspective_projection(
        self,
        fov_y: float,
        width: int,
        height: int,
        near: float = 1.0,
        far: float = 100.0,
    ):
        """
        Returns:
            P: Perspective projection matrix, (4, 4)
                P = [
                        [2*n/(r-l), 0.0,        (r+l)/(r-l),    0.0         ],
                        [0.0,       2*n/(t-b),  (t+b)/(t-b),    0.0         ],
                        [0.0,       0.0,        -(f+n)/(f-n),   -(f*n)/(f-n)],
                        [0.0,       0.0,        -1.0,            0.0         ]
                    ]
        """
        deg2rad = math.pi / 180
        tan_fov_y = math.tan(fov_y * 0.5 * deg2rad)
        tan_fov_x = tan_fov_y * (width / height)
        top = tan_fov_y * near
        bottom = -top
        right = tan_fov_x * near
        left = -right
        z_sign = -1.0

        proj = torch.zeros([4, 4], device=self.device)

        proj[0, 0] = 2.0 * near / (right - left)
        proj[1, 1] = 2.0 * near / (top - bottom)
        proj[0, 2] = (right + left) / (right - left)
        proj[1, 2] = (top + bottom) / (top - bottom)
        proj[3, 2] = z_sign
        proj[2, 2] = z_sign * (far + near) / (far - near)
        proj[2, 3] = -(2.0 * far * near) / (far - near)
        return proj

    def intrinsics_perspective_projection(
        self,
        K: torch.Tensor,
        width: int,
        height: int,
        scale: float = 1.0,
        near: float = 0.01,
        far: float = 100.0,
    ):
        """
        Transform points from camera space (x: right, y: up, z: out) to clip space (x: right, y: up, z: in)

        For information check out the math:
        https://www.songho.ca/opengl/gl_projectionmatrix.html

        Args:
            K: Intrinsic matrix, (3, 3)
                K = [
                        [fx, 0, cx],
                        [0, fy, cy],
                        [0,  0,  1],
                    ]
        Returns:
            P: Perspective projection matrix, (4, 4)
                P = [
                        [2*fx/w, 0.0,     (w - 2*cx)/w,             0.0                     ],
                        [0.0,    2*fy/h,  (h - 2*cy)/h,             0.0                     ],
                        [0.0,    0.0,     -(far+near) / (far-near), -2*far*near / (far-near)],
                        [0.0,    0.0,     -1.0,                     0.0                     ]
                    ]
        """
        w = width
        h = height
        fx = K[0, 0] * scale
        fy = K[1, 1] * scale
        cx = K[0, 2] * scale
        cy = K[1, 2] * scale

        proj = torch.zeros([4, 4], device=self.device)
        proj[0, 0] = fx * 2 / w
        proj[1, 1] = fy * 2 / h
        proj[0, 2] = (w - 2 * cx) / w
        proj[1, 2] = (h - 2 * cy) / h
        proj[2, 2] = -(far + near) / (far - near)
        proj[2, 3] = -2 * far * near / (far - near)
        proj[3, 2] = -1
        return proj

    def convert_to_homo_coords(self, p: torch.Tensor):
        shape = list(p.shape)
        assert shape[-1] == 3
        shape[-1] = shape[-1] + 1
        p_homo = torch.ones(shape, device=p.device)
        p_homo[..., :3] = p
        return p_homo

    def clip_transform(self, p_camera: torch.Tensor):
        return torch.matmul(p_camera, self.projection_matrix.T)

    def ndc_transform(self, p_camera: torch.Tensor):
        p_clip = self.clip_transform(p_camera)
        p_clip[..., 0] /= p_clip[..., 3]
        p_clip[..., 1] /= p_clip[..., 3]
        p_clip[..., 2] /= p_clip[..., 3]
        p_clip[..., 3] /= p_clip[..., 3]
        return p_clip

    def xy_ndc_to_screen(self, xy_ndc: torch.Tensor):
        u = (xy_ndc[..., 0] + 1) * 0.5 * self.width
        # this is the flip because ndc is (-1,-1) bottom left we want that the scrren
        # starts top left (0, 0), hence we can index the image with the coordiante
        v = (1 - xy_ndc[..., 1]) * 0.5 * self.height
        return torch.stack([u, v], dim=-1)

    def screen_transform(self, p_camera: torch.Tensor):
        p_ndc = self.ndc_transform(p_camera)
        depth = p_camera[..., 2]
        uv = self.xy_ndc_to_screen(p_ndc)
        return torch.stack([uv[..., 0], uv[..., 1], depth], dim=-1)

    def unproject_points(self, xy_depth: torch.Tensor):
        """
        The x,y data contains the values in ndc coordinates space, hence they
        are between (-1, 1) and the depth is the value from the z-plane in
        camera space, hence the sign should be negative.

        xy_depth[i] = [x[i], y[i], depth[i]]
        """
        z_camera = xy_depth[..., 2]
        p1 = self.projection_matrix[2, 2]
        p2 = self.projection_matrix[2, 3]
        p3 = self.projection_matrix[3, 2]
        w_clip = p3 * z_camera
        z_ndc = (p1 * z_camera + p2) / w_clip
        p_ndc = torch.stack([xy_depth[..., 0], xy_depth[..., 1], z_ndc], dim=-1)
        p_ndc = self.convert_to_homo_coords(p_ndc)
        # convert back to clip space
        p_clip = p_ndc
        p_clip[..., 0] *= w_clip
        p_clip[..., 1] *= w_clip
        p_clip[..., 2] *= w_clip
        p_clip[..., 3] *= w_clip
        p_clip = torch.nan_to_num(p_clip, nan=0.0)
        # extract only x,y,z component
        p_camera = torch.matmul(p_clip, self.projection_matrix.inverse().T)
        return p_camera[..., :3]

    def depth_map_transform(self, depth: torch.Tensor):
        """
        Transforms a depth map to camera coordinates, where the depth values
        are positive, which is flipped in this function, because +Z points
        towards the camera.
        """
        assert depth.shape[0] == self.original_height
        assert depth.shape[1] == self.original_width

        # calculate the mask with the backgound and then output the forground, e.g the resize is
        # downing the interpolation in a way that by boolen we just keep true.
        size = (self.height, self.width)
        b_mask = v2.functional.resize((depth == 0).unsqueeze(0), size=size).squeeze(0)
        depth = v2.functional.resize(depth.unsqueeze(0), size=size).squeeze(0)

        depth_camera = -depth.unsqueeze(-1).to(self.device)
        x_ndc = torch.linspace(-1, 1, steps=self.width, device=self.device)
        y_ndc = torch.linspace(1, -1, steps=self.height, device=self.device)
        y_grid, x_grid = torch.meshgrid(y_ndc, x_ndc, indexing="ij")
        xy_ndc = torch.stack([x_grid, y_grid], dim=-1)
        xy_depth = torch.concatenate([xy_ndc, depth_camera], dim=-1)
        p_camera = self.unproject_points(xy_depth=xy_depth)

        p_camera[b_mask, :] = 0.0
        mask = ~b_mask

        return p_camera, mask

    def to(self, device: str = "cuda"):
        self.projection_matrix = self.projection_matrix.to(device)
        return self


def camera2normal(point: torch.Tensor):
    """Calculate the normal image from the camera image.

    We calculate the normal in camera space, hence we also need to normalize with the
    depth information. Note that the normal is basically on which direction we have the
    stepest decent.

    More can be be found here:
    https://stackoverflow.com/questions/34644101/

    They have (-dz/dx,-dz/dy,1), however we are in camera space hence we need to
    calculate the gradient in pixel space, e.g. also the delta x and delta y are in
    camera space.

    Args:
        depth (torch.Tensor): Camera image of dim (B, H, W, 3).

    Returns:
        (torch.Tensor): The normal image based on the depth/camera of dim (B, H, W, 3).
    """
    # NOTE in order to calc that only with depth image, we need to make sure that the
    # depth is in pixel space.
    point = point.clone()

    # make sure that on the boundary is nothing wrong calculated
    point[point.sum(-1) == 0] = torch.nan  # some large value

    _, H, W, _ = point.shape
    normals = torch.ones_like(point)
    normals *= -1  # make sure that the default normal looks to the camera

    x_right = torch.arange(2, W)
    x_left = torch.arange(0, W - 2)
    dzx = point[:, :, x_right, 2] - point[:, :, x_left, 2]
    dx = point[:, :, x_right, 0] - point[:, :, x_left, 0]
    normals[:, :, 1:-1, 0] = dzx / dx

    y_right = torch.arange(2, H)
    y_left = torch.arange(0, H - 2)
    dzy = point[:, y_right, :, 2] - point[:, y_left, :, 2]
    dy = point[:, y_right, :, 1] - point[:, y_left, :, 1]
    normals[:, 1:-1, :, 1] = dzy / dy

    # normalized between [-1, 1] and remove artefacs
    normals = normals / torch.norm(normals, dim=-1).unsqueeze(-1)
    normals = torch.nan_to_num(normals, 0)
    normals[:, :1, :, :] = 0
    normals[:, -1:, :, :] = 0
    normals[:, :, :1, :] = 0
    normals[:, :, -1:, :] = 0

    mask = normals.sum(-1) != 0

    return normals, mask
