import torch
from torch import nn


class Residuals(nn.Module):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    ####################################################################################
    # Loss and Jacobian Computation
    ####################################################################################

    @property
    def names(self):
        return [self.name]

    def forward(self, **kwargs):
        raise NotImplementedError()

    def step(self, **kwargs):
        residuals = self.forward(**kwargs)  # (C,)
        F = torch.cat(residuals)
        info = {n: r for n, r in zip(self.names, residuals)}
        return F.reshape(-1), info  # (C,)


####################################################################################
# ChainedLoss
####################################################################################


class ChainedResiduals(Residuals):
    def __init__(self, chain: dict[str, Residuals] = {}):
        super().__init__()
        self.chain = chain

    @property
    def names(self):
        return [f.name for f in self.chain.values()]

    def forward(self, **kwargs):
        residuals = []
        for f in self.chain.values():
            residuals.extend(f(**kwargs))
        return residuals


####################################################################################
# Dense Loss Terms
####################################################################################
class Point2PlaneResiduals(Residuals):
    name: str = "point2plane"

    def forward(self, **kwargs):
        s_point = kwargs["s_point"]
        t_point = kwargs["t_point"]
        t_normal = kwargs["t_normal"]
        weights = kwargs.get("weights")
        residuals = ((s_point - t_point) * t_normal).sum(-1)
        if weights is not None:
            return [self.weight * residuals * weights]
        return [self.weight * residuals]


class Point2PointResiduals(Residuals):
    name: str = "point2point"

    def forward(self, **kwargs):
        s_point = kwargs["s_point"]
        t_point = kwargs["t_point"]
        weights = kwargs.get("weights")
        residuals = s_point - t_point
        if weights is not None:
            return [self.weight * residuals * weights]
        return [self.weight * residuals]


class SymmetricICPResiduals(Residuals):
    name: str = "symmetricICP"

    def forward(self, **kwargs):
        s_point = kwargs["s_point"]
        t_point = kwargs["t_point"]
        s_normal = kwargs["s_normal"]
        t_normal = kwargs["t_normal"]
        weights = kwargs.get("weights")
        residuals = ((s_point - t_point) * (s_normal + t_normal)).sum(-1)
        if weights is not None:
            return [self.weight * residuals * weights]
        return [self.weight * residuals]


####################################################################################
# Regularization Loss Terms
####################################################################################


class ShapeRegularizationResiduals(Residuals):
    name: str = "shape_regularization"

    def forward(self, **kwargs):
        params = kwargs["params"]
        shape_params = params["shape_params"]
        if shape_params is not None:
            return [self.weight * shape_params.view(-1)]
        device = kwargs["s_point"].device
        return [torch.tensor([], device=device)]


class ExpressionRegularizationResiduals(Residuals):
    name: str = "expression_regularization"

    def forward(self, **kwargs):
        params = kwargs["params"]
        expression_params = params["expression_params"]
        if expression_params is not None:
            return [self.weight * expression_params.view(-1)]
        device = kwargs["s_point"].device
        return [torch.tensor([], device=device)]


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
        return [self.weight * (t_vertices - s_vertices)]


####################################################################################
# Deep Features
####################################################################################


class FeatureResiduals(Residuals):
    name: str = "feature"

    def forward(self, **kwargs):
        t_vertices = kwargs["t_feature"]
        s_vertices = kwargs["s_feature"]
        return [self.weight(t_vertices - s_vertices)]
