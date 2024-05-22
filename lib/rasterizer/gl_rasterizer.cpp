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

    const char *vShaderCode = "#version 460 core\n" STRINGIFY_SHADER_SOURCE(
        layout(location = 0) in vec4 aPos;

        void main() {
            gl_Position = aPos;
        });

    const char *gShaderCode = "#version 460 core\n" STRINGIFY_SHADER_SOURCE(
        layout(triangles) in;
        layout(triangle_strip, max_vertices = 3) out;

        void main() {
            gl_Position = gl_in[0].gl_Position;
            EmitVertex();
            gl_Position = gl_in[1].gl_Position;
            EmitVertex();
            gl_Position = gl_in[2].gl_Position;
            EmitVertex();
            EndPrimitive();
        });

    const char *fShaderCode = "#version 460 core\n" STRINGIFY_SHADER_SOURCE(
        out vec4 FragColor;
        void main() {
            FragColor = vec4(1.0, 0.0, 0.0, 1.0);
        });
    Shader shader(vShaderCode, gShaderCode, fShaderCode);

    // creates the vertex attribute object and the vertex buffer object
    // this should be probabiliyt be better without copying when using cuda or so.
    unsigned int VAO, VBO, EBO;
    GL_CHECK_ERROR(glGenVertexArrays(1, &VAO));
    GL_CHECK_ERROR(glGenBuffers(1, &VBO));
    GL_CHECK_ERROR(glGenBuffers(1, &EBO));

    GL_CHECK_ERROR(glBindVertexArray(VAO));
    GL_CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, VBO));

    GL_CHECK_ERROR(glBufferData(GL_ARRAY_BUFFER, vertices.numel() * sizeof(float), vertices.data_ptr<float>(), GL_STATIC_DRAW));
    GL_CHECK_ERROR(glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0));

    GL_CHECK_ERROR(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO));
    GL_CHECK_ERROR(glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.numel() * sizeof(uint32_t), indices.data_ptr<uint32_t>(), GL_STATIC_DRAW));

    GL_CHECK_ERROR(glEnableVertexAttribArray(0));
    GL_CHECK_ERROR(glBindVertexArray(0));

    // rasterizes the vertices using opengl
    shader.use();
    GL_CHECK_ERROR(glClearColor(0.0f, 1.0f, 1.0f, 1.0f));
    GL_CHECK_ERROR(glClear(GL_COLOR_BUFFER_BIT));
    GL_CHECK_ERROR(glBindVertexArray(VAO));
    GL_CHECK_ERROR(glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0));

    // read the output into a tensor again
    std::vector<unsigned char> pixels(eglData.pbufferWidth * eglData.pbufferHeight * 4); // RGBA format
    glReadPixels(0, 0, eglData.pbufferWidth, eglData.pbufferHeight, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    torch::Tensor out = readImageFromOpenGL(eglData.pbufferWidth, eglData.pbufferHeight);

    // destroy the context
    destroyEGL(eglData);

    // return the tensor
    return out;
}
