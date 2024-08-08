import logging

import lightning as L
import torch

from lib.model.flame.flame import Flame
from lib.optimizer.base import BaseOptimizer
from lib.optimizer.residuals import Residuals
from lib.renderer.renderer import Renderer

log = logging.getLogger()


class FlameOptimizer(L.LightningModule):
    def __init__(
        self,
        # models
        flame: Flame,
        correspondence_module: torch.nn.Module,
        weighting_module: torch.nn.Module,
        # renderer settings
        renderer: Renderer,
        # optimization settings
        residuals: Residuals,
        optimizer: BaseOptimizer,
        max_iters: int = 1,
        max_optims: int = 1,
    ):
        super().__init__()

        # models
        self.flame = flame
        self.c_module = correspondence_module
        self.w_module = weighting_module

        # dfferentiable renderer
        self.renderer = renderer

        # optimization settings
        self.optimizer = optimizer
        self.residuals = residuals
        self.max_iters = max_iters
        self.max_optims = max_optims

    def forward(
        self,
        s_point: torch.Tensor,
        s_normal: torch.Tensor,
        s_mask: torch.Tensor,
        params: dict[str, torch.Tensor],
    ):
        self.optimizer.set_params(params)
        for iter_step in range(self.max_iters):
            # render the current state of the model
            m_out = self.flame(**self.optimizer.get_params())
            r_out = self.renderer.render_full(
                vertices=m_out["vertices"],  # (B, V, 3)
                faces=self.flame.faces,  # (F, 3)
            )
            t_mask = r_out["mask"]
            t_point = r_out["point"]
            t_normal = r_out["normal"]

            # establish correspondences
            c_out = self.c_module.predict(
                s_point=s_point,
                s_normal=s_normal,
                t_point=t_point,
                t_normal=t_normal,
            )
            mask = self.c_module.mask(s_mask=s_mask, t_mask=t_mask, **c_out)
            t_new_point = self.c_module.transform(t_value=t_point, **c_out)

            # predict weights
            weights = self.w_module(s_point=s_point, t_point=t_new_point)

            def residual_closure(*args):
                # differentiable rendering without rasterization
                new_params = self.optimizer.residual_params(args)
                m_out = self.flame(**new_params)

                # recompute to perform interpolation of the point inside closure
                t_point = self.renderer.interpolate(
                    vertices_idx=r_out["vertices_idx"],
                    bary_coords=r_out["bary_coords"],
                    attributes=m_out["vertices"],
                )
                t_new_point = self.c_module.transform(t_value=t_point, **c_out)

                F = self.residuals.step(
                    mask=mask,
                    weights=weights,
                    s_normal=s_normal,
                    s_point=s_point,
                    t_normal=t_normal,
                    t_point=t_new_point,
                    params=new_params,
                )
                return F, F

            # inner optimization loop
            for optim_step in range(self.max_optims):
                self.optimizer.step(residual_closure)
        return self.optimizer.get_params()


class VertexOptimizer(L.LightningModule):
    def __init__(
        self,
        flame: Flame,
        optimizer: BaseOptimizer,
        residuals: Residuals,
        max_iters: int = 1,
    ):
        super().__init__()
        self.flame = flame
        self.optimizer = optimizer
        self.residuals = residuals
        self.max_iters = max_iters

    def forward(self, s_vertices: torch.Tensor, params: dict[str, torch.Tensor]):
        def residual_closure(*args):
            new_params = self.optimizer.residual_params(args)
            m_out = self.flame(**new_params)
            F = self.residuals.step(
                t_vertices=m_out["vertices"],
                s_vertices=s_vertices,
            )
            return F, F

        self.optimizer.set_params(params)
        for iter_step in range(self.max_iters):
            self.optimizer.step(residual_closure)
        return self.optimizer.get_params()
