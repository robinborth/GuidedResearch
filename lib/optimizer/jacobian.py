import torch

from lib.model.flame import FLAME
from lib.model.loss import calculate_point2plane


class FlameJacobian:
    def __init__(
        self,
        model: FLAME,
        batch: dict,
        correspondences: dict,
        params: list[str],
    ):
        # save the shape and frame idxs
        self.shape_idx = batch["shape_idx"][0]
        # check that in one batch we only have one person
        assert (self.shape_idx == batch["shape_idx"]).all()
        self.frame_idx = batch["frame_idx"]

        # settings
        self.model = model
        self.batch = batch
        self.correspondences = correspondences
        self.B = self.frame_idx.shape[0]

        # the full list of params, the ORDER here is IMPORTANT, because that
        # defines how the jacobians matrix is build.
        shared_params = {
            "shape_params": model.shape_params.weight.shape[-1],
        }
        unique_params = {
            "expression_params": model.expression_params.weight.shape[-1],
            "global_pose": model.global_pose.weight.shape[-1],
            "transl": model.transl.weight.shape[-1],
            "neck_pose": model.neck_pose.weight.shape[-1],
            "eye_pose": model.eye_pose.weight.shape[-1],
            "jaw_pose": model.jaw_pose.weight.shape[-1],
        }
        # filter the params for the ones that are used for the optimization
        self.shared_params = {k: v for k, v in shared_params.items() if k in params}
        self.unique_params = {k: v for k, v in unique_params.items() if k in params}
        self.params: list[str] = [
            *self.shared_params.keys(),
            *self.unique_params.keys(),
        ]

        # sizes for helping building the jacobian
        self.sN = sum(self.shared_params.values())  # the dim of the shared params
        self.uN = sum(self.unique_params.values())  # the dim of ONE unqiue params
        self.N = self.sN + self.uN * self.B  # the total dim of the jacobian NxN

    def _jacobian_unknowns_preperation(self, **kwargs):
        # the state for the params that are not optimized
        state = {}

        # initilize the jacobian inputs with the correct order
        jacobian_input = {}
        for param in self.params:
            jacobian_input[param] = None

        # fill the jacobian input
        for key, value in kwargs.items():
            if key in self.params:
                jacobian_input[key] = value
            else:
                state[key] = value

        # preprocess the jacobian input and remove the None
        jacobian_input = [v for v in jacobian_input.values() if v is not None]
        assert len(jacobian_input) == len(self.params)

        return tuple(jacobian_input), state

    def _flame_dict_input(
        self,
        jacobian_input: list[torch.Tensor],
        state: dict[str, torch.Tensor],
    ):
        flame_params = {}
        for p_name, param in zip(self.params, jacobian_input):
            flame_params[p_name] = param[None]
        for p_name, param in state.items():
            flame_params[p_name] = param[None]
        return flame_params

    def build(self):
        """Build the jacobian matrix.

        The jacobian matrix is block-dense, only shape is shared between batches.
        The final dimension is (m x n) where m is the number of total residuals and
        n is the number of unknowns. The general structure is like the following:

        residual 0  : shape, exp_0, pose_0, exp_1, pose_1, ..., exp_n, pose_n
        residual 1  : shape, exp_0, pose_0, exp_1, pose_1, ..., exp_n, pose_n
        residual ...: shape, exp_0, pose_0, exp_1, pose_1, ..., exp_n, pose_n
        residual r1 : shape, exp_0, pose_0, exp_1, pose_1, ..., exp_n, pose_n
        residual 0  : shape, exp_0, pose_0, exp_1, pose_1, ..., exp_n, pose_n
        residual 1  : shape, exp_0, pose_0, exp_1, pose_1, ..., exp_n, pose_n
        residual ...: shape, exp_0, pose_0, exp_1, pose_1, ..., exp_n, pose_n
        residual r2 : shape, exp_0, pose_0, exp_1, pose_1, ..., exp_n, pose_n
        ...
        residual 0  : shape, exp_0, pose_0, exp_1, pose_1, ..., exp_n, pose_n
        residual 1  : shape, exp_0, pose_0, exp_1, pose_1, ..., exp_n, pose_n
        residual ...: shape, exp_0, pose_0, exp_1, pose_1, ..., exp_n, pose_n
        residual rm : shape, exp_0, pose_0, exp_1, pose_1, ..., exp_n, pose_n

        Further note that pose is the following concatenation:
        pose: global_pose, transl, neck_pose, eye_pose, jaw_pose

        However, we only compute the jacobian of the unknowns that are optimized.

        Args:
            batch (dict): The input batch.
            correspondences (dict): The correspondences.

        Returns:
            torch.Tensor: The jacobian matrix.
        """
        # tuple parameters
        batch = self.batch
        correspondences = self.correspondences
        flame_input = self.model.flame_input_dict(batch)

        # compute the jacobians
        jacobians = []
        for idx in range(self.B):
            # state
            mask = correspondences["mask"][idx]
            q = batch["point"][idx][mask]  # (R, 3)
            n = correspondences["normal"][idx][mask]  # (R, 3)
            vertices_idx = correspondences["vertices_idx"][idx]
            bary_coords = correspondences["bary_coords"][idx]

            # jacobian input
            _flame_input = {k: v[idx] for k, v in flame_input.items()}
            x, state = self._jacobian_unknowns_preperation(**_flame_input)

            def closure(*args):
                flame_params = self._flame_dict_input(jacobian_input=args, state=state)
                vertices = self.model.forward(**flame_params)  # (B, V, 3)
                p = self.model.renderer.mask_interpolate(
                    vertices_idx=vertices_idx[None],
                    bary_coords=bary_coords[None],
                    attributes=vertices,
                    mask=mask[None],
                )  # (R, 3)  (residuals for the current batch)
                point2plane = calculate_point2plane(q=q, p=p, n=n)  # (R,)
                return point2plane

            jacobian = torch.autograd.functional.jacobian(
                func=closure,
                inputs=x,
                create_graph=False,  # this is currently note differentiable
                strategy="forward-mode",
                vectorize=True,
            )
            jacobians.append(jacobian)

        # build the empty jacobiabn
        N = self.N
        M = sum([j[0].shape[0] for j in jacobians])  # number of residuals
        J = torch.zeros((M, N), device=self.model.device)

        # fill the block-dense jacobian
        r_offset = 0  # residual offset
        for batch_idx, jacs in enumerate(jacobians):
            # extract the residual size
            sM = jacs[0].shape[0]
            for jac, param in zip(jacs, self.params):
                i, j = self.offset(p_name=param, batch_idx=batch_idx)
                J[r_offset : r_offset + sM, i:j] = jac
            # add the residual offset
            r_offset += sM

        return J

    def offset(self, p_name: str, batch_idx: int):
        # find the offset of the jacobian
        oS = [0]
        for k, v in self.shared_params.items():
            oS.append(oS[-1] + v)
            if k == p_name:
                break

        oU = [0]
        for k, v in self.unique_params.items():
            oU.append(oU[-1] + v)
            if k == p_name:
                break

        # unique params, added offset of the shared
        if p_name in self.unique_params:
            offset = self.sN + self.uN * batch_idx
            return offset + oU[-2], offset + oU[-1]

        # simple shared offset
        return oS[-2], oS[-1]

    def loss(self, fx: torch.Tensor):
        """Calculates the loss of the model based on the params"""
        batch = self.batch
        correspondences = self.correspondences

        # update the flame flame input
        flame_params = self.model.flame_input_dict(batch)
        for p_name in self.params:
            param: list[torch.Tensor] = []
            for b_idx in range(self.B):
                i, j = self.offset(p_name, b_idx)
                param.append(fx[i:j])
            flame_params[p_name] = torch.stack(param, dim=0)

        mask = correspondences["mask"]
        vertices = self.model.forward(**flame_params)
        p = self.model.renderer.mask_interpolate(
            vertices_idx=correspondences["vertices_idx"],
            bary_coords=correspondences["bary_coords"],
            attributes=vertices,
            mask=mask,
        )  # (C, 3)
        q = batch["point"][mask]  # (C, 3)
        n = correspondences["normal"][mask]  # (C, 3)
        point2plane = calculate_point2plane(q=q, p=p, n=n)  # (C,)
        return point2plane.mean()
