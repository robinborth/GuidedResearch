#include <EGL/egl.h>
#include <GL/gl.h>
#include <GLES3/gl3.h>

#include <torch/extension.h>

#include <iostream>
#include "shader.h"
#include "egl_utils.h"
#include "utils.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Function to read the current image from OpenGL into a torch::Tensor
torch::Tensor readImageFromOpenGL(int pbufferWidth, int pbufferHeight)
{
    // Allocate memory to store the pixel data
    std::vector<unsigned char> pixels(pbufferWidth * pbufferHeight * 4); // RGBA format

    // Use glReadPixels to read the pixel data from the framebuffer
    glReadPixels(0, 0, pbufferWidth, pbufferHeight, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());

    // Copy the pixel data from the vector into a torch::Tensor
    // Note: Assuming RGBA format
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    return torch::from_blob(pixels.data(), {pbufferHeight, pbufferWidth, 4}, options).clone();
}

torch::Tensor rasterize(torch::Tensor vertices, int width, int height)
{
    // init the context
    EGLContextData eglData;
    eglData.pbufferHeight = height;
    eglData.pbufferWidth = width;
    initEGL(eglData);

    // vertex shader modifies the 3d vertex positions, e.g. performe here perspective divide
    const char *vShaderCode = "#version 460 core\n" STRINGIFY_SHADER_SOURCE(
        layout(location = 0) in vec3 aPos;

        void main() {
            gl_Position = vec4(aPos, 1.0);
        });
    // fragment shader do here compute the color value
    const char *fShaderCode = "#version 460 core\n" STRINGIFY_SHADER_SOURCE(
        out vec4 FragColor;
        void main() {
            FragColor = vec4(1.0, 0.0, 0.0, 1.0);
        });
    Shader shader(vShaderCode, fShaderCode);

    // creates the vertex attribute object and the vertex buffer object
    // this should be probabiliyt be better without copying when using cuda or so.
    unsigned int VAO, VBO;
    glGenBuffers(1, &VBO);
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    // TODO we should check that the type of the tensor is indeed a float32
    glBufferData(GL_ARRAY_BUFFER, vertices.numel() * sizeof(float), vertices.data_ptr<float>(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);

    // rasterizes the vertices using opengl
    shader.use();
    glClearColor(0.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    // read the output into a tensor again
    std::vector<unsigned char> pixels(eglData.pbufferWidth * eglData.pbufferHeight * 4); // RGBA format
    glReadPixels(0, 0, eglData.pbufferWidth, eglData.pbufferHeight, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    torch::Tensor out = readImageFromOpenGL(eglData.pbufferWidth, eglData.pbufferHeight);

    // destroy the context
    destroyEGL(eglData);

    // return the tensor
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("rasterize", &rasterize, "OpenGL Rasterizer");
}