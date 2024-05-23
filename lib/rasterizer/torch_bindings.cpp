#include <torch/extension.h>

// forward declarations of the torch bindings that are exportet
torch::Tensor rasterize(torch::Tensor vertices, torch::Tensor indices, torch::Tensor normals, int width, int height, int cuda_device_idx);

// define the modules that are exportet
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("rasterize", &rasterize, "OpenGL Rasterizer");
}