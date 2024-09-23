import torch
from torch import nn


class Residuals(nn.Module):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

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


class RegularizationResiduals(Residuals):
    def __init__(self, name: str, weight: float = 1.0):
        super().__init__()
        self.weight = weight
        self.name = name

    def forward(self, **kwargs):
        params = kwargs["params"][self.name]
        if params is not None:
            return [self.weight * params.view(-1)]
        device = kwargs["s_point"].device
        return [torch.tensor([], device=device)]


class NeuralRegularizationResiduals(Residuals):
    def __init__(self, name: str, weight: float = 1.0):
        super().__init__()
        self.weight = weight
        self.name = name

    def forward(self, **kwargs):
        params = kwargs["params"][self.name]
        delta_params = kwargs["regularize"][self.name]
        residuals = (params - delta_params).view(-1)
        residuals = residuals * residuals
        return [self.weight * residuals]


####################################################################################
# Sparse Keypoints
####################################################################################


class LandmarkResiduals(Residuals):
    name: str = "landmark"

    def forward(self, **kwargs):
        mask = kwargs["s_landmark_mask"]
        s_landmark = kwargs["s_landmark"]
        t_landmark = kwargs["t_landmark"]
        residuals = (s_landmark[mask] - t_landmark[mask]).flatten()
        return [self.weight * residuals]


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
