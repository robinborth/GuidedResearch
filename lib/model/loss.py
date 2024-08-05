import time
from typing import Any

import torch
from torch import nn
from torch.autograd.functional import jacobian
from torch.func import jacfwd, jacrev, vmap

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


def regularization_distance(params: list[torch.Tensor]):
    latents = [(p**2).view(-1) for p in params if p is not None]
    if latents:
        return torch.cat(latents)
    return torch.tensor(0.0)


class BaseLoss(nn.Module):
    def __init__(
        self,
        model: FLAME,
        batch: dict,
        weight_map: torch.Tensor,
        correspondences: dict,
        params: list[torch.Tensor],
        p_names: list[str],
        logger: Any,
        jacobian_fn: str = "v2",
        weight: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.batch = batch
        self.weight_map = weight_map
        self.correspondences = correspondences
        self.params = params
        self.p_names = p_names
        self.weight = weight
        self.logger = logger
        self.jacobian_fn = jacobian_fn

    ####################################################################################
    # Loss and Jacobian Computation
    ####################################################################################
    def forward(self, out: dict[str, torch.Tensor]):
        raise NotImplementedError

    def model_step(self):
        return self.model.model_step(
            batch=self.batch,
            correspondences=self.correspondences,
            params=self.params,
            p_names=self.p_names,
        )

    def _residual_step(
        self,
        model: FLAME,
        batch: dict,
        correspondences: dict,
        params: list[torch.Tensor],
        p_names: list[str],
    ):
        self.logger.time_tracker.start("model_step")
        out = model.model_step(batch, correspondences, params, p_names)
        self.logger.time_tracker.start("loss_step", stop=True)
        loss = self.weight * self.forward(out)  # (C,)
        self.logger.time_tracker.stop()
        return loss.reshape(-1)  # (C,)

    def loss_step(self):
        residuals = self._residual_step(
            model=self.model,
            batch=self.batch,
            correspondences=self.correspondences,
            params=self.params,
            p_names=self.p_names,
        )

        # for regularization we don't have a residual and just return the empty tensor
        if not residuals.numel():
            return {"loss": residuals, self.name: residuals}

        loss = residuals**2
        return {"loss": loss, self.name: loss}

    def _jacobian_stepv1(
        self,
        model: FLAME,
        batch: dict,
        correspondences: dict,
        params: list[torch.Tensor],
        p_names: list[str],
    ):
        def closure(*args):
            return self._residual_step(model, batch, correspondences, args, p_names)

        jacobians = torch.autograd.functional.jacobian(
            func=closure,
            inputs=tuple(params),
            create_graph=False,  # this is currently note differentiable
            strategy="forward-mode",
            vectorize=True,
        )

        J = torch.cat([j.flatten(-2) for j in jacobians], dim=-1)  # (M, N)
        F = self._residual_step(model, batch, correspondences, params, p_names)
        return J, F

    def _residual_stepv2(
        self,
        model: FLAME,
        batch: dict,
        weight_map: torch.Tensor,
        correspondences: dict,
        params: list[torch.Tensor],
        p_names: list[str],
    ):
        self.logger.time_tracker.start("model_step")
        out = model.model_step(batch, correspondences, params, p_names)
        self.logger.time_tracker.start("loss_step", stop=True)
        w_map = weight_map[correspondences["mask"]]
        loss = w_map * self.weight * self.forward(out)  # (C,)
        self.logger.time_tracker.stop()
        return loss.reshape(-1)  # (C,)

    def _jacobian_stepv2(
        self,
        model: FLAME,
        batch: dict,
        weight_map: torch.Tensor,
        correspondences: dict,
        params: list[torch.Tensor],
        p_names: list[str],
    ):
        def closure(*args):
            F = self._residual_stepv2(
                model,
                batch,
                weight_map,
                correspondences,
                args,
                p_names,
            )
            return F, F

        jacobian_fn = jacfwd(
            func=closure,
            argnums=tuple(range(len(params))),
            has_aux=True,
        )
        jacobians, F = jacobian_fn(*params)
        J = torch.cat([j.flatten(-2) for j in jacobians], dim=-1)  # (M, N)
        return J, F

    def _residual_stepv3(
        self,
        model: FLAME,
        batch: dict,
        correspondences: dict,
        params: torch.Tensor,
        p_names: list[str],
    ):
        self.logger.time_tracker.start("model_step")
        out = model.model_stepv3(batch, correspondences, params, p_names)
        self.logger.time_tracker.start("loss_step", stop=True)
        loss = self.weight * self.forward(out)  # (C,)
        self.logger.time_tracker.stop()
        return loss.reshape(-1)  # (C,)

    def _jacobian_stepv3(
        self,
        model: FLAME,
        batch: dict,
        correspondences: dict,
        params: list[torch.Tensor],
        p_names: list[str],
    ):
        def closure(arg):
            F = self._residual_stepv3(model, batch, correspondences, arg, p_names).sum()
            return F, F

        _params = torch.cat([p.flatten(-2) for p in params], dim=-1)  # (M, N)
        jacobian_fn = jacfwd(func=closure, has_aux=True)
        J, F = jacobian_fn(_params)
        return J, F

    def jacobian_step(self):
        if self.jacobian_fn == "v1":
            return self._jacobian_stepv1(
                model=self.model,
                batch=self.batch,
                correspondences=self.correspondences,
                params=self.params,
                p_names=self.p_names,
            )
        elif self.jacobian_fn == "v2":
            return self._jacobian_stepv2(
                model=self.model,
                batch=self.batch,
                correspondences=self.correspondences,
                weight_map=self.weight_map,
                params=self.params,
                p_names=self.p_names,
            )
        elif self.jacobian_fn == "v3":
            return self._jacobian_stepv3(
                model=self.model,
                batch=self.batch,
                correspondences=self.correspondences,
                params=self.params,
                p_names=self.p_names,
            )
        raise ValueError("No jacobian_fn version that matches!")

    ####################################################################################
    # Closures
    ####################################################################################

    def loss_closure(self):
        return lambda: self.loss_step()["loss"].sum()  # sum of the squared residuals

    def jacobian_closure(self):
        return lambda: self.jacobian_step()


####################################################################################
# ChainedLoss
####################################################################################


class ChainedLoss(BaseLoss):
    def __init__(self, chain: dict = {}, **kwargs):
        super().__init__(**kwargs)
        self.chain: dict[str, BaseLoss] = {}
        for loss, weight in chain.items():
            if loss == "point2plane":
                _loss: BaseLoss = Point2PlaneLoss(weight=weight, **kwargs)
            elif loss == "point2point":
                _loss = Point2PointLoss(weight=weight, **kwargs)
            elif loss == "shape_regularization":
                _loss = ShapeRegularizationLoss(weight=weight, **kwargs)
            elif loss == "expression_regularization":
                _loss = ExpressionRegularizationLoss(weight=weight, **kwargs)
            elif loss == "symmetricICP":
                _loss = SymmetricICPLoss(weight=weight, **kwargs)
            elif loss == "landmark2d":
                _loss = Landmark2DLoss(weight=weight, **kwargs)
            elif loss == "landmark3d":
                _loss = Landmark3DLoss(weight=weight, **kwargs)
            else:
                raise ValueError(f"Value {loss} is not specified.")
            self.chain[loss] = _loss

    def loss_step(self):
        loss = {}
        for key, l_func in self.chain.items():
            loss[key] = l_func.loss_step()["loss"]
        loss["loss"] = torch.cat(tuple(loss.values()))
        return loss

    def jacobian_step(self):
        J = []
        F = []
        for l_func in self.chain.values():
            _J, _F = l_func.jacobian_step()
            J.append(_J)
            F.append(_F)
        J = torch.cat(J, dim=0)
        F = torch.cat(F, dim=0)
        return J, F


####################################################################################
# Dense Loss Terms
####################################################################################
class Point2PlaneLoss(BaseLoss):
    name: str = "point2plane"

    def forward(self, out: dict[str, torch.Tensor]):
        point = out["point"]
        point_gt = out["point_gt"]
        normal = out["normal"]
        plane_dist = ((point - point_gt) * normal).sum(-1)
        return plane_dist


class Point2PointLoss(BaseLoss):
    name: str = "point2point"

    def forward(self, out: dict[str, torch.Tensor]):
        point = out["point"]
        point_gt = out["point_gt"]
        return point - point_gt


class SymmetricICPLoss(BaseLoss):
    name: str = "symmetricICP"

    def forward(self, out: dict[str, torch.Tensor]):
        point = out["point"]
        point_gt = out["point_gt"]
        normal = out["normal"]
        normal_gt = out["normal_gt"]
        plane_dist = ((point - point_gt) * (normal + normal_gt)).sum(-1)
        return plane_dist


####################################################################################
# Regularization Loss Terms
####################################################################################


class ShapeRegularizationLoss(BaseLoss):
    name: str = "shape_regularization"

    def forward(self, out: dict[str, torch.Tensor]):
        shape_params = out["shape_params"]
        if shape_params is not None:
            return shape_params.view(-1)

        device = out["point"].device
        return torch.tensor([], device=device)


class ExpressionRegularizationLoss(BaseLoss):
    name: str = "expression_regularization"

    def forward(self, out: dict[str, torch.Tensor]):
        expression_params = out["expression_params"]
        if expression_params is not None:
            return expression_params.view(-1)

        device = out["point"].device
        return torch.tensor([], device=device)


####################################################################################
# Sparse Keypoints
####################################################################################


class Landmark2DLoss(BaseLoss):
    name: str = "landmark2d"

    def forward(self, out: dict[str, torch.Tensor]):
        lm_2d_ndc = out["lm_2d_ndc"]
        lm_2d_ndc_gt = out["lm_2d_ndc_gt"]
        return lm_2d_ndc - lm_2d_ndc_gt


class Landmark3DLoss(BaseLoss):
    name: str = "landmark3d"

    def forward(self, out: dict[str, torch.Tensor]):
        lm_3d_camera = out["lm_3d_camera"]
        lm_3d_camera_gt = out["lm_3d_camera_gt"]
        return lm_3d_camera - lm_3d_camera_gt
