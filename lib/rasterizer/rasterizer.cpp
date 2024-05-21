#include <torch/extension.h>

#include <EGL/egl.h>
#include <GL/gl.h>
#include <GLES3/gl3.h>

// // #define STB_IMAGE_IMPLEMENTATION
// // #include <stb_image.h>
// // #define STB_IMAGE_WRITE_IMPLEMENTATION
// // #include "stb_image_write.h"

#include <iostream>
#include "shader.h"
#include "egl_utils.h"

// #include <test.h>
// #include <vector>

// float vertices[] = {
//     -0.5f, -0.5f, 0.0f,
//     0.5f, -0.5f, 0.0f,
//     0.0f, 0.5f, 0.0f};

void rasterize()
{
    EGLContextData eglData;
    initEGL(eglData);
    // handle here the case when the initilization did not work.

    // vertex shader modifies the 3d vertex positions, e.g. performe here perspective divide
    const char *vShaderCode = "#version 330 core\n" STRINGIFY_SHADER_SOURCE(
        layout(location = 0) in vec3 aPos;

        void main() {
            gl_Position = vec4(aPos, 1.0);
        });

    // fragment shader do here compute the color value
    const char *fShaderCode = "#version 330 core\n" STRINGIFY_SHADER_SOURCE(
        out vec4 FragColor;
        void main() {
            FragColor = vec4(1.0, 0.0, 0.0, 1.0);
        });

    // defines the shader that creates a program links the shader codes together
    // Shader shader(vShaderCode, fShaderCode);
    std::cout << "rasterize" << std::endl;
    destroyEGL(eglData);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("rasterize", &rasterize, "OpenGL Rasterizer");
}