import torch


def distance(vertices: torch.Tensor, points: torch.Tensor):
    """Calculates the distance between all vertiices and full point cloud.

    Args:
        vertices (torch.Tensor): The vertices in world-space of dim (B, V, 3)
        points (torch.Tensor): The points in world-space of dim (B, P, 3)

    Returns:
        (torch.Tensor): The min distance of dim (B, V, P)
    """
    sx = torch.sum(vertices**2, dim=-1).unsqueeze(-1)  # (B, V, 1)
    sy = torch.sum(points**2, dim=-1).unsqueeze(-2)  # (B, 1, P)
    sxy = torch.bmm(vertices, points.transpose(-2, -1))  # (B, V, P)
    dist_in = torch.nn.functional.relu(sx - 2 * sxy + sy)
    return torch.sqrt(dist_in)  # (B, V, P)


def point_to_point_full(vertices: torch.Tensor, points: torch.Tensor):
    """Calculates the point-to-point distance between vertiices and a point cloud.

    Args:
        vertices (torch.Tensor): The vertices in world-space of dim (B, V, 3)
        points (torch.Tensor): The points in world-space of dim (B, P, 3)

    Returns:
        (torch.Tensor): The min distance of dim (B, V)
    """
    dist = distance(vertices, points)  # (B, V, P)
    return dist.min(-1).values  # (B, V)


def chamfer_distance(vertices: torch.Tensor, points: torch.Tensor):
    """Calculates the point-to-point distance between vertiices and a point cloud.

    Args:
        vertices (torch.Tensor): The vertices in world-space of dim (B, V, 3)
        points (torch.Tensor): The points in world-space of dim (B, P, 3)

    Returns:
        (torch.Tensor): The min distance of dim (B, V)
    """
    dist = distance(vertices, points)  # (B, V, P)
    return dist.min(-1).values.mean(-1) + dist.min(-2).values.mean(-1)  # (B,)


def landmark_3d_distance(landmarks: torch.Tensor, gt_landmarks: torch.Tensor):
    """Calculates the 3d landmark loss between the vertices.

    Args:
        landmarks: (torch.Tensor): The 3d coords of the flame landmarks (105, 3)
        gt_landmarks: (torch.Tensor): The 3d coords of the mediapipe landmarks (105, 3)
    """
    return torch.norm(landmarks - gt_landmarks, dim=-1)


def landmark_2d_distance(landmarks: torch.Tensor, gt_landmarks: torch.Tensor):
    """Calculates the 2d landmark loss between the vertices."""
    return torch.norm(landmarks - gt_landmarks, dim=-1)


def calculate_point2point(q: torch.Tensor, p: torch.Tensor):
    return torch.pow(q - p, 2).sum(-1)  # (B, W, H)


def calculate_point2plane(q: torch.Tensor, p: torch.Tensor, n: torch.Tensor):
    return torch.pow(((q - p) * n).sum(-1), 2)  # (B, W, H)
