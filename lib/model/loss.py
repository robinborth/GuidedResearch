import torch

from lib.model.utils import bary_coord_interpolation


def point_to_point_loss(vertices: torch.Tensor, points: torch.Tensor):
    """Calculates the point-to-point distance between vertiices and a point cloud.

    Args:
        vertices (torch.Tensor): The vertices in world-space of dim (B, V, 3)
        points (torch.Tensor): The points in world-space of dim (B, P, 3)

    Returns:
        (torch.Tensor): The min distance of dim (B, V)
    """
    sx = torch.sum(vertices**2, dim=-1).unsqueeze(-1)  # (B, V, 1)
    sy = torch.sum(points**2, dim=-1).unsqueeze(-2)  # (B, 1, P)
    sxy = torch.bmm(vertices, points.transpose(-2, -1))  # (B, V, P)
    dist_in = torch.nn.functional.relu(sx - 2 * sxy + sy)
    dist = torch.sqrt(dist_in)  # (B, V, P)
    return dist.min(-1).values  # (B, V)


def chamfer_distance_loss(vertices: torch.Tensor, points: torch.Tensor):
    """Calculates the point-to-point distance between vertiices and a point cloud.

    Args:
        vertices (torch.Tensor): The vertices in world-space of dim (B, V, 3)
        points (torch.Tensor): The points in world-space of dim (B, P, 3)

    Returns:
        (torch.Tensor): The min distance of dim (B, V)
    """
    sx = torch.sum(vertices**2, dim=-1).unsqueeze(-1)  # (B, V, 1)
    sy = torch.sum(points**2, dim=-1).unsqueeze(-2)  # (B, 1, P)
    sxy = torch.bmm(vertices, points.transpose(-2, -1))  # (B, V, P)
    dist_in = torch.nn.functional.relu(sx - 2 * sxy + sy)
    dist = torch.sqrt(dist_in)  # (B, V, P)
    dist_v = dist.min(-1).values  # (B, V)
    dist_p = dist.min(-2).values  # (B, P)
    return dist_v.mean(-1) + dist_p.mean(-1)  # (B,)


def landmark_3d_loss(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    bary_coords: torch.Tensor,
    mediapipe_lm3d: torch.Tensor,
    lm_mediapipe_idx: torch.Tensor,
):
    """Calculates the 3d landmark loss between the vertices.

    Args:
        vertices (torch.Tensor): The vertices in camera coordinate system (V, 3)
        faces (torch.Tensor): The indexes of the vertices, e.g. the faces (105, 3)
        bary_coords (torch.Tensor): The interpolation coeficients of dim (105, 3)
        lm_mediapipe_idx: (torch.Tensor): The idx of the mediapipe landmarks (105,)
        mediapipe_lm3d: (torch.Tensor): The idx of the mediapipe landmarks (478, 3)

    Returns:
        (torch.Tensor): Vertices with the attributes barycentric interpolated (F, D)
    """
    vertices_lm3d = bary_coord_interpolation(
        faces=faces,
        attributes=vertices,
        bary_coords=bary_coords,
    )  # (105, 3)
    gt_lm3d = mediapipe_lm3d[lm_mediapipe_idx]  # (105, 3)
    return torch.norm(vertices_lm3d - gt_lm3d, dim=-1), vertices_lm3d
