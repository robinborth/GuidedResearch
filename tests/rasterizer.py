import torch
from torchvision.transforms import v2

from lib.rasterizer import rasterize

width = 800
height = 600
vertices = torch.tensor(
    data=[-0.5, -0.5, 0.0, 0.5, -0.5, 0.0, 0.0, 0.5, 0.0],
    dtype=torch.float32,
)
out = rasterize(vertices=vertices, width=width, height=height)

image = v2.functional.to_pil_image(out.permute(2, 0, 1))
image.save("temp/output.png")
