import torch
from torch import nn


class Residuals(nn.Module):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    ####################################################################################
    # Loss and Jacobian Computation
    ####################################################################################

    def forward(self, **kwargs):
        raise NotImplementedError()

    def step(self, **kwargs):
        F = self.weight * self.forward(**kwargs)  # (C,)
        return F.reshape(-1)  # (C,)


####################################################################################
# ChainedLoss
####################################################################################


class ChainedResiduals(Residuals):
    def __init__(self, chain: list[Residuals] = [], weight: float = 1.0):
        super().__init__(weight=weight)
        self.chain = chain

    def forward(self, **kwargs):
        return torch.cat([f(**kwargs) for f in self.chain])


####################################################################################
# Dense Loss Terms
####################################################################################
class Point2PlaneResiduals(Residuals):
    name: str = "point2plane"

    def forward(self, **kwargs):
        s_point = kwargs["s_point"]
        t_point = kwargs["t_point"]
        t_normal = kwargs["t_normal"]
        residuals = ((s_point - t_point) * t_normal).sum(-1)
        if weights := kwargs.get("weights"):
            return residuals * weights
        return residuals


class Point2PointResiduals(Residuals):
    name: str = "point2point"

    def forward(self, **kwargs):
        s_point = kwargs["s_point"]
        t_point = kwargs["t_point"]
        residuals = s_point - t_point
        if weights := kwargs.get("weights"):
            return residuals * weights
        return residuals


class SymmetricICPResiduals(Residuals):
    name: str = "symmetricICP"

    def forward(self, **kwargs):
        s_point = kwargs["s_point"]
        t_point = kwargs["t_point"]
        s_normal = kwargs["s_normal"]
        t_normal = kwargs["t_normal"]
        residuals = ((s_point - t_point) * (s_normal + t_normal)).sum(-1)
        if weights := kwargs.get("weights"):
            return residuals * weights
        return residuals


####################################################################################
# Regularization Loss Terms
####################################################################################


class ShapeRegularizationResiduals(Residuals):
    name: str = "shape_regularization"

    def forward(self, **kwargs):
        params = kwargs["params"]
        shape_params = params["shape_params"]
        if shape_params is not None:
            return shape_params.view(-1)
        device = kwargs["s_point"].device
        return torch.tensor([], device=device)


class ExpressionRegularizationResiduals(Residuals):
    name: str = "expression_regularization"

    def forward(self, **kwargs):
        params = kwargs["params"]
        expression_params = params["expression_params"]
        if expression_params is not None:
            return expression_params.view(-1)
        device = kwargs["s_point"].device
        return torch.tensor([], device=device)


####################################################################################
# Sparse Keypoints
####################################################################################


class Landmark2DResiduals(Residuals):
    name: str = "landmark2d"


class Landmark3DResiduals(Residuals):
    name: str = "landmark3d"


####################################################################################
# Direct Correspondences
####################################################################################


class VertexResiduals(Residuals):
    name: str = "vertex"

    def forward(self, **kwargs):
        t_vertices = kwargs["t_vertices"]
        s_vertices = kwargs["s_vertices"]
        return t_vertices - s_vertices


####################################################################################
# Deep Features
####################################################################################


class FeatureResiduals(Residuals):
    name: str = "feature"

    def forward(self, **kwargs):
        t_vertices = kwargs["t_feature"]
        s_vertices = kwargs["s_feature"]
        return t_vertices - s_vertices
