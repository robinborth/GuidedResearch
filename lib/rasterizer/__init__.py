import importlib

from torch.utils.cpp_extension import load

# # loads the rasterizer module in cpp
plugin_name = "rasterizer_plugin"
rasterizer = load(
    name=plugin_name,
    sources=[
        "lib/rasterizer/rasterizer.cpp",
        "lib/rasterizer/egl_utils.cpp",
        "lib/rasterizer/utils.cpp",
        "lib/rasterizer/shader.cpp",
    ],
    extra_cflags=["-g"],
    extra_ldflags=["-lEGL", "-lGL"],
    verbose=True,
)
plugin = importlib.import_module(plugin_name)


def rasterize():
    """
    The interface of the python function.
    """
    plugin.rasterize()


__all__ = ["rasterize"]
