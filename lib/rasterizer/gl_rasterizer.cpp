#include "gl_utils.h"
#include "gl_context.h"
#include "gl_shader.h"
#include "torch_utils.h"

#include <GLES3/gl3.h>
#include <iostream>

torch::Tensor rasterize(torch::Tensor vertices, torch::Tensor indices, int width, int height)
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
    unsigned int VAO, VBO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    glBufferData(GL_ARRAY_BUFFER, vertices.numel() * sizeof(float), vertices.data_ptr<float>(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);

    // unsigned int _indices[] = {
    //     0, 1, 3,
    //     1, 2, 3};
    // std::cout << "_indices: size" << sizeof(_indices) << std::endl;
    // std::cout << "torch: size" << indices.numel() * sizeof(uint32_t) << std::endl;

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.numel() * sizeof(uint32_t), indices.data_ptr<uint32_t>(), GL_STATIC_DRAW);

    // glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    // glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(_indices), _indices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glBindVertexArray(0);

    // rasterizes the vertices using opengl
    shader.use();
    glClearColor(0.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    // read the output into a tensor again
    std::vector<unsigned char> pixels(eglData.pbufferWidth * eglData.pbufferHeight * 4); // RGBA format
    glReadPixels(0, 0, eglData.pbufferWidth, eglData.pbufferHeight, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    torch::Tensor out = readImageFromOpenGL(eglData.pbufferWidth, eglData.pbufferHeight);

    // destroy the context
    destroyEGL(eglData);

    // return the tensor
    return out;
}
