import math
from pathlib import Path

import torch
from torchvision.transforms import v2

from lib.model.flame import FLAME
from lib.rasterizer import Rasterizer
from lib.renderer.renderer import Renderer
from lib.utils.mesh import weighted_vertex_normals

width = 800
height = 800

root_folder = Path(__file__).parent.parent
flame_dir = str((root_folder / "checkpoints/flame2023").resolve())
data_dir = str((root_folder / "data/dphm_christoph_mouthmove").resolve())

flame = FLAME(flame_dir=flame_dir, data_dir=data_dir)
flame.init_params(
    global_pose=[0, 0, 0],
    transl=[0.0, 0.0, -0.5],
)
# vertices in clip space (B, V, 3)
vertices, _ = flame()


def convert_to_homo_coords(coords):
    shape = list(coords.shape)
    shape[-1] = shape[-1] + 1
    homo_coords = torch.ones(shape, device=vertices.device)
    homo_coords[:, :, :3] = coords
    return homo_coords


def fov_perspective_projection(
    fov: float,
    aspect: float,
    near: float = 1.0,
    far: float = 100.0,
):
    scale = math.tan(fov * 0.5 * math.pi / 180) * near
    r = aspect * scale
    l = -r
    t = scale
    b = -t
    return torch.tensor(
        [
            [2 * near / (r - l), 0.0, (r + l) / (r - l), 0.0],
            [0.0, 2 * near / (t - b), (t + b) / (t - b), 0.0],
            [0.0, 0.0, -(far + near) / (far - near), -(2 * far * near) / (far - near)],
            [0.0, 0.0, -1.0, 0.0],
        ]
    )


verts_homo = convert_to_homo_coords(vertices)
M = fov_perspective_projection(fov=36, aspect=(width / height))
verts_homo = torch.matmul(verts_homo, M)


# indices of the faces (F, 3)
faces = flame.faces
# faces = flame.masked_faces(vertices)

# make this more robust when not working on the gpu
device = "cuda"
print("Cuda device index: ", torch.cuda.current_device())
print("Input device:", device)

print("Input Vertices: ")
print("shape: ", verts_homo.shape)
print("min: ", verts_homo.min())
print("max: ", verts_homo.max())
print("mean: ", verts_homo.mean())

print("Input indices: ")
print("shape: ", faces.shape)
print("min: ", faces.min())
print("max: ", faces.max())

# for now we just output the normals
renderer = Renderer(width=width, height=height, device=device)
vertex_normals = weighted_vertex_normals(vertices, faces)
normal, mask = renderer.render(
    vertices=verts_homo.to(device),
    faces=faces.to(device),
    attributes=vertex_normals.to(device),
)
normal_image = (((normal + 1) / 2) * 255).to(torch.uint8)
normal_image[~mask] = 0

# save multiple images per batch
for idx, img in enumerate(normal_image):
    image = v2.functional.to_pil_image(img.permute(2, 0, 1))
    path = root_folder / f"temp/flame/output{idx}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
