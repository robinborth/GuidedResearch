import math
from pathlib import Path

import torch
from torchvision.transforms import v2

from lib.model.flame import FLAME
from lib.rasterizer import Rasterizer
from lib.renderer.renderer import Renderer

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
M = fov_perspective_projection(fov=45, aspect=(width / height))
vertices = torch.matmul(verts_homo, M)


# indices of the faces (F, 3)
# indices = flame.faces
indices = flame.masked_faces(vertices)

# make this more robust when not working on the gpu
device = "cuda"
print("Cuda device index: ", torch.cuda.current_device())
print("Input device:", device)

print("Input Vertices: ")
print("shape: ", vertices.shape)
print("min: ", vertices.min())
print("max: ", vertices.max())
print("mean: ", vertices.mean())

print("Input indices: ")
print("shape: ", indices.shape)
print("min: ", indices.min())
print("max: ", indices.max())

# for now we just output the normals
rasterizer = Rasterizer(width=width, height=height)
fragments = rasterizer.rasterize(
    vertices=vertices.to(device),
    indices=indices.to(device),
)
print("Output device:", fragments.bary_coords.device)

print("bary_coords:")
print("shape: ", fragments.bary_coords.shape)
print("min: ", fragments.bary_coords.min())
print("max: ", fragments.bary_coords.max())
print("mean: ", fragments.bary_coords.mean())

print("pix_to_face:")
print("shape: ", fragments.pix_to_face.shape)
print("min: ", fragments.pix_to_face.min())
print("max: ", fragments.pix_to_face.max())

# save multiple images per batch
for idx, bc in enumerate(fragments.bary_coords):
    image = v2.functional.to_pil_image(bc.permute(2, 0, 1))
    path = root_folder / f"temp/flame/output{idx}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
