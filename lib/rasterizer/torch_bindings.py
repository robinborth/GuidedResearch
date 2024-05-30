import importlib
from dataclasses import dataclass

import torch
from torch.utils.cpp_extension import load

# # loads the rasterizer module in cpp
plugin_name = "rasterizer_plugin"
rasterizer = load(
    name=plugin_name,
    sources=[
        "lib/rasterizer/gl_context.cpp",
        "lib/rasterizer/gl_rasterizer.cpp",
        "lib/rasterizer/gl_shader.cpp",
        "lib/rasterizer/gl_utils.cpp",
        "lib/rasterizer/torch_bindings.cpp",
        "lib/rasterizer/torch_utils.cpp",
    ],
    extra_cflags=["-g"],
    extra_ldflags=["-lEGL", "-lGL"],
    verbose=True,
)
plugin = importlib.import_module(plugin_name)


@dataclass
class Fragments:
    pix_to_face: torch.Tensor
    bary_coords: torch.Tensor


def rasterize(
    vertices: torch.Tensor,
    indices: torch.Tensor,
    width: int,
    height: int,
    cuda_device_idx: int = -1,
) -> Fragments:
    """
    The interface of the python function.
    """
    f = plugin.rasterize(vertices, indices, width, height, cuda_device_idx)
    return Fragments(pix_to_face=f.pix_to_face, bary_coords=f.bary_coords)
