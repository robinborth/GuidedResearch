from pathlib import Path

import torch
from torchvision.transforms import v2

from lib.model.flame import FLAME
from lib.rasterizer import Rasterizer
from lib.renderer.camera import FoVCamera
from lib.renderer.renderer import Renderer
from lib.utils.mesh import vertex_normals

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


# indices of the faces (F, 3)
faces = flame.faces
# faces = flame.masked_faces(vertices)

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
print("shape: ", faces.shape)
print("min: ", faces.min())
print("max: ", faces.max())

# for now we just output the normals
renderer = Renderer(width=width, height=height, device=device)
normals = vertex_normals(vertices, faces)
normal, mask = renderer.render(
    vertices=vertices.to(device),
    faces=faces.to(device),
    attributes=normals.to(device),
)
normal_image = (((normal + 1) / 2) * 255).to(torch.uint8)
normal_image[~mask] = 0

# save multiple images per batch
for idx, img in enumerate(normal_image):
    image = v2.functional.to_pil_image(img.permute(2, 0, 1))
    path = root_folder / f"temp/flame/output{idx}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
