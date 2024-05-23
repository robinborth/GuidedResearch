import importlib

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


def rasterize(
    vertices: torch.Tensor,
    indices: torch.Tensor,
    width: int,
    height: int,
    cuda_device_idx: int = -1,
):
    """
    The interface of the python function.
    """
    return plugin.rasterize(vertices, indices, width, height, cuda_device_idx)
