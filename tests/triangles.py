from pathlib import Path

import torch
from torchvision.transforms import v2

from lib.rasterizer import Rasterizer

root_folder = Path(__file__).parent.parent

width = 800
height = 600

# vertices in clip space (B, V, 3)
vertices = torch.tensor(
    data=[
        [
            [0.5, 0.5, 0.0, 1.0],
            [0.5, -0.5, 0.0, 1.0],
            [-0.5, -0.5, 0.0, 1.0],
            [-0.5, 0.5, 0.5, 1.0],
        ],
        [
            [0.9, 0.5, 0.0, 1.0],
            [0.1, -0.7, 0.0, 1.0],
            [-0.4, -0.5, 0.0, 1.0],
            [-0.5, 0.2, 0.0, 1.0],
        ],
    ],
    dtype=torch.float32,
)

# indices of the faces (F, 3)
indices = torch.tensor(
    data=[
        [0, 1, 3],
        [1, 2, 3],
        [0, 2, 3],
    ],
    dtype=torch.int64,
)

# make this more robust when not working on the gpu
device = "cuda"
print("Cuda device index: ", torch.cuda.current_device())
print("Input device:", device)
print("Input vertices shape: ", vertices.shape)
print("Input indices shape: ", indices.shape)

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
    path = root_folder / f"temp/triangles/output{idx}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
