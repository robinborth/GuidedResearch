import torch
import torch.nn as nn

from lib.flame.lbs import lbs
from lib.flame.utils import load_flame


class FLAME(nn.Module):
    """
    Given flame parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 3D facial landmarks
    """

    def __init__(
        self,
        flame_dir: str,
        shape_params: int = 300,
        expression_params: int = 100,
    ):
        super().__init__()

        # load the face model
        flame_model = load_flame(flame_dir=flame_dir, return_tensors="pt")

        self.faces = nn.Parameter(flame_model["f"], requires_grad=False)
        self.v_template = nn.Parameter(flame_model["v_template"])
        self.shapedirs = nn.Parameter(flame_model["shapedirs"])
        self.J_regressor = nn.Parameter(flame_model["J_regressor"])
        self.lbs_weights = nn.Parameter(flame_model["weights"])

        # Fixing remaining Shape betas, where there are total 300 shape parameters to
        # control FLAME; But one can use the first few parameters to express the shape.
        default_shape = torch.zeros(300 - shape_params, requires_grad=False)
        self.shape_betas = nn.Parameter(default_shape, requires_grad=False)

        # Fixing remaining expression betas, where there are total 100 shape expression
        # parameters to control FLAME; But one can use the first few parameters to
        # express the expression
        default_exp = torch.zeros(100 - expression_params, requires_grad=False)
        self.expression_betas = nn.Parameter(default_exp, requires_grad=False)

        # Eyeball and neck rotation
        default_global_pose = torch.zeros(3, requires_grad=False)
        self.global_pose = nn.Parameter(default_global_pose, requires_grad=False)
        default_neck_pose = torch.zeros(3, requires_grad=False)
        self.neck_pose = nn.Parameter(default_neck_pose, requires_grad=False)
        default_jaw_pose = torch.zeros(3, requires_grad=False)
        self.jaw_pose = nn.Parameter(default_jaw_pose, requires_grad=False)
        default_eyball_pose = torch.zeros(6, requires_grad=False)
        self.eye_pose = nn.Parameter(default_eyball_pose, requires_grad=False)

        # Fixing 3D translation since we use translation in the image plane
        default_transl = torch.zeros(3, requires_grad=False)
        self.transl = nn.Parameter(default_transl, requires_grad=False)

        # Pose blend shape basis
        # TODO why is the dimension (5023, 3, 36)
        num_pose_basis = flame_model["posedirs"].shape[-1]
        posedirs = torch.reshape(flame_model["posedirs"], [-1, num_pose_basis]).T
        self.posedirs = nn.Parameter(posedirs)

        # indices of parents for each joints
        parents = flame_model["kintree_table"][0]
        parents[0] = -1
        self.parents = nn.Parameter(parents, requires_grad=False)

    def forward(
        self,
        shape_params=None,
        expression_params=None,
        global_pose=None,
        neck_pose=None,
        jaw_pose=None,
        eye_pose=None,
        transl=None,
    ):
        """
        Input:
            shape_params: number of shape parameters
            expression_params: number of expression parameters
            global_pose: number of global pose parameters (3)
            neck_pose: number of neck pose parameters (3)
            jaw_pose: number of jaw pose parameters (3)
            eye_pose: number of eye pose parameters (6)
        return:
            vertices: V X 3
        """
        # create the betas merged with shape and expression
        shape = torch.cat([shape_params, self.shape_betas])
        expression = torch.cat([expression_params, self.expression_betas])
        betas = torch.cat([shape, expression])

        # create the pose merged with global, jaw, neck and left/right eye
        global_pose = global_pose if global_pose is not None else self.global_pose
        jaw_pose = eye_pose if eye_pose is not None else self.jaw_pose
        neck_pose = neck_pose if neck_pose is not None else self.neck_pose
        eye_pose = eye_pose if eye_pose is not None else self.eye_pose
        pose = torch.cat([global_pose, neck_pose, jaw_pose, eye_pose], dim=-1)

        # apply the linear blend skinning model
        vertices, _ = lbs(
            betas.unsqueeze(0),
            pose.unsqueeze(0),
            self.v_template.unsqueeze(0),
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
        )
        vertices = vertices[0]

        # add the translation to the vertices and offsets
        transl = transl if transl is not None else self.transl
        vertices += transl

        return vertices
