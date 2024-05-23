#include <torch/extension.h>
#include "gl_rasterizer.h"

// define the modules that are exportet
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("rasterize", &rasterize, "OpenGL Rasterizer");
}