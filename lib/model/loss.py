import torch


def point_to_point(vertices: torch.Tensor, points: torch.Tensor):
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
    dist = torch.sqrt(sx - 2 * sxy + sy)  # (B, V, P)
    return dist.min(-1).values  # (B, V)
