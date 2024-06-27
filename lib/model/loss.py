import torch
from torch import nn

from lib.model.flame import FLAME


def distance(vertices: torch.Tensor, points: torch.Tensor, max_points: int = 5000):
    """Calculates the distance between all vertiices and full point cloud.

    Args:
        vertices (torch.Tensor): The vertices in world-space of dim (B, V, 3)
        points (torch.Tensor): The points in world-space of dim (B, P, 3)

    Returns:
        (torch.Tensor): The min distance of dim (B, V, P)
    """
    sx_idx = torch.randperm(vertices.shape[1])[:max_points].to(vertices.device)
    sx = torch.sum(vertices[:, sx_idx] ** 2, dim=-1).unsqueeze(-1)  # (B, V, 1)
    sy_idx = torch.randperm(points.shape[1])[:max_points].to(points.device)
    sy = torch.sum(points[:, sy_idx] ** 2, dim=-1).unsqueeze(-2)  # (B, 1, P)
    sxy = torch.bmm(
        vertices[:, sx_idx], points[:, sy_idx].transpose(-2, -1)
    )  # (B, V, P)
    dist_in = torch.nn.functional.relu(sx - 2 * sxy + sy)
    return torch.sqrt(dist_in)  # (B, V, P)


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


def point2point_distance(q: torch.Tensor, p: torch.Tensor):
    return torch.sqrt(torch.pow(q - p, 2).sum(-1))  # (B, W, H)


def point2plane_distance(q: torch.Tensor, p: torch.Tensor, n: torch.Tensor):
    return torch.sqrt(torch.pow(((q - p) * n).sum(-1), 2))  # (B, W, H)


class BaseLoss(nn.Module):
    def __init__(
        self,
        model: FLAME,
        batch: dict,
        correspondences: dict,
        params: list[torch.Tensor],
        p_names: list[str],
        weight: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.batch = batch
        self.correspondences = correspondences
        self.params = params
        self.p_names = p_names
        self.weight = weight

    ####################################################################################
    # Loss and Jacobian Computation
    ####################################################################################
    def forward(self, out: dict[str, torch.Tensor]):
        raise NotImplementedError

    def _loss_step(
        self,
        model: FLAME,
        batch: dict,
        correspondences: dict,
        params: list[torch.Tensor],
        p_names: list[str],
    ):
        out = model.model_step(batch, correspondences, params, p_names)
        loss = self.weight * self.forward(out)  # (C,)
        return loss.reshape(-1)  # (B, C)

    def loss_step(self):
        return self._loss_step(
            model=self.model,
            batch=self.batch,
            correspondences=self.correspondences,
            params=self.params,
            p_names=self.p_names,
        )

    def _jacobian_step(
        self,
        model: FLAME,
        batch: dict,
        correspondences: dict,
        params: list[torch.Tensor],
        p_names: list[str],
    ):
        def closure(*args):
            return self._loss_step(model, batch, correspondences, args, p_names)

        jacobians = torch.autograd.functional.jacobian(
            func=closure,
            inputs=tuple(params),
            create_graph=False,  # this is currently note differentiable
            strategy="forward-mode",
            vectorize=True,
        )
        J = torch.cat([j.flatten(-2) for j in jacobians], dim=-1)  # (M, N)
        return J

    def jacobian_step(self):
        return self._jacobian_step(
            model=self.model,
            batch=self.batch,
            correspondences=self.correspondences,
            params=self.params,
            p_names=self.p_names,
        )

    ####################################################################################
    # Closures
    ####################################################################################

    def loss_closure(self):
        return lambda: self.loss_step().mean()  # this returns a scalar

    def jacobian_closure(self):
        return lambda: self.jacobian_step()


####################################################################################
# ChainedLoss
####################################################################################


class ChainedLoss(BaseLoss):
    def __init__(self, chain: dict = {}, **kwargs):
        super().__init__(**kwargs)
        self.chain: list[BaseLoss] = []
        for loss, weight in chain.items():
            if loss == "point2plane":
                self.chain.append(Point2PlaneLoss(weight=weight, **kwargs))
            elif loss == "point2point":
                self.chain.append(Point2PointLoss(weight=weight, **kwargs))
            elif loss == "regularization":
                self.chain.append(LatentRegularizationLoss(weight=weight, **kwargs))
            elif loss == "symmetricICP":
                self.chain.append(SymmetricICPLoss(weight=weight, **kwargs))
            elif loss == "landmark2d":
                self.chain.append(Landmark2DLoss(weight=weight, **kwargs))
            elif loss == "landmark3d":
                self.chain.append(Landmark3DLoss(weight=weight, **kwargs))
            else:
                raise ValueError(f"Value {loss} is not specified.")

    def loss_step(self):
        loss = [l_func.loss_step() for l_func in self.chain]
        return torch.cat(loss)

    def jacobian_step(self):
        J = [l_func.jacobian_step() for l_func in self.chain]
        return torch.cat(J, dim=0)


####################################################################################
# Dense Loss Terms
####################################################################################
class Point2PlaneLoss(BaseLoss):
    def forward(self, out: dict[str, torch.Tensor]):
        point = out["point"]
        point_gt = out["point_gt"]
        normal = out["normal"]
        plane_dist = ((point - point_gt) * normal).sum(-1)
        return plane_dist**2


class Point2PointLoss(BaseLoss):
    def forward(self, out: dict[str, torch.Tensor]):
        point = out["point"]
        point_gt = out["point_gt"]
        return (point - point_gt) ** 2


class SymmetricICPLoss(BaseLoss):
    def forward(self, out: dict[str, torch.Tensor]):
        point = out["point"]
        point_gt = out["point_gt"]
        normal = out["normal"]
        normal_gt = out["normal_gt"]
        plane_dist = ((point - point_gt) * (normal + normal_gt)).sum(-1)
        return plane_dist**2


####################################################################################
# Regularization Loss Terms
####################################################################################


class LatentRegularizationLoss(BaseLoss):
    def forward(self, out: dict[str, torch.Tensor]):
        device = out["point"].device
        shape_params = out["shape_params"]
        expression_params = out["expression_params"]

        params = []
        if shape_params is not None:
            params.append(shape_params.view(-1))
        if expression_params is not None:
            params.append(expression_params.view(-1))

        if not params:
            return torch.tensor([], device=device)

        latents = torch.cat(params)
        return latents**2


####################################################################################
# Sparse Keypoints
####################################################################################


class Landmark2DLoss(BaseLoss):
    def forward(self, out: dict[str, torch.Tensor]):
        lm_2d_ndc = out["lm_2d_ndc"]
        lm_2d_ndc_gt = out["lm_2d_ndc_gt"]
        return (lm_2d_ndc - lm_2d_ndc_gt) ** 2


class Landmark3DLoss(BaseLoss):
    def forward(self, out: dict[str, torch.Tensor]):
        lm_3d_camera = out["lm_3d_camera"]
        lm_3d_camera_gt = out["lm_3d_camera_gt"]
        return (lm_3d_camera - lm_3d_camera_gt) ** 2
