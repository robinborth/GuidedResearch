#include <torch/extension.h>
#include "gl_rasterizer.h"
#include "gl_context.h"

// define the modules that are exportet
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    pybind11::class_<Fragments>(m, "Fragments")
        .def(pybind11::init<>())
        .def_readwrite("pix_to_face", &Fragments::pix_to_face)
        .def_readwrite("bary_coords", &Fragments::bary_coords);
    pybind11::class_<GLContext>(m, "GLContext")
        .def(pybind11::init<int, int, int>())
        .def_readwrite("display", &GLContext::display)
        .def_readwrite("context", &GLContext::context)
        .def_readwrite("width", &GLContext::width)
        .def_readwrite("height", &GLContext::height)
        .def_readwrite("cudaDeviceIdx", &GLContext::cudaDeviceIdx)
        .def("destroy", &GLContext::destroy);
    m.def("rasterize", &rasterize, "OpenGL Rasterizer");
}