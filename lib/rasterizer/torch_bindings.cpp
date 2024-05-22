#include <torch/extension.h>

// forward declarations of the torch bindings that are exportet
torch::Tensor rasterize(torch::Tensor vertices, torch::Tensor indices, int width, int height);

// define the modules that are exportet
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("rasterize", &rasterize, "OpenGL Rasterizer");
}