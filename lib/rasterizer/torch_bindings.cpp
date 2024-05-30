#include <torch/extension.h>
#include "gl_rasterizer.h"

// define the modules that are exportet
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    pybind11::class_<Fragments>(m, "Fragments")
        .def(pybind11::init<>())
        .def_readwrite("pix_to_face", &Fragments::pix_to_face)
        .def_readwrite("bary_coords", &Fragments::bary_coords);
    m.def("rasterize", &rasterize, "OpenGL Rasterizer");
}