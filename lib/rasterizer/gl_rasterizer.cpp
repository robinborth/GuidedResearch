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

        out VS_OUT {
            vec3 color;
            vec3 bary;
        } vs_out;

        void main() {
            gl_Position = aPos;
            vs_out.color = vec3(1.0, 0.0, 0.0);
        });

    const char *gShaderCode = "#version 460 core\n" STRINGIFY_SHADER_SOURCE(
        layout(triangles) in;
        layout(triangle_strip, max_vertices = 3) out;

        in VS_OUT {
            vec3 color;
            vec3 bary;
        } gs_in[];

        out VS_OUT {
            vec3 color;
            vec3 bary;
        } gs_out;

        void main() {
            gs_out.color = gs_in[0].color;
            gs_out.bary = vec3(1.0, 0.0, 0.0);
            gl_Position = gl_in[0].gl_Position;
            EmitVertex();

            gs_out.color = gs_in[1].color;
            gs_out.bary = vec3(0.0, 1.0, 0.0);
            gl_Position = gl_in[1].gl_Position;
            EmitVertex();

            gs_out.color = vec3(0.0, 1.0, 1.0);
            gs_out.bary = vec3(0.0, 0.0, 1.0);
            gl_Position = gl_in[2].gl_Position;
            EmitVertex();

            EndPrimitive();
        });

    const char *fShaderCode = "#version 460 core\n" STRINGIFY_SHADER_SOURCE(
        layout(location = 0) out vec4 gColor;
        layout(location = 1) out vec4 gPosition;

        in VS_OUT {
            vec3 color;
            vec3 bary;
        } fs_in;

        void main() {
            gColor = vec4(fs_in.color, 1.0);
            gPosition = vec4(1.0, 0.0, 0.0, 1.0);
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

    unsigned int gBuffer;
    glGenFramebuffers(1, &gBuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, gBuffer);
    unsigned int gColor, gPosition;

    // - color gbuffer
    GL_CHECK_ERROR(glGenTextures(1, &gColor));
    GL_CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, gColor));
    GL_CHECK_ERROR(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, eglData.pbufferWidth, eglData.pbufferHeight, 0, GL_RGBA, GL_FLOAT, nullptr));
    GL_CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
    GL_CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
    GL_CHECK_ERROR(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gColor, 0));

    GL_CHECK_ERROR(glGenTextures(1, &gPosition));
    GL_CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, gPosition));
    GL_CHECK_ERROR(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, eglData.pbufferWidth, eglData.pbufferHeight, 0, GL_RGBA, GL_FLOAT, nullptr));
    GL_CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
    GL_CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
    GL_CHECK_ERROR(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, gPosition, 0));

    unsigned int attachments[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
    GL_CHECK_ERROR(glDrawBuffers(2, attachments));

    // Check if the framebuffer is complete
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        std::cout << "ERROR FRAMEBUFFER NOT COMPLETE" << std::endl;
    }

    // rasterizes the vertices using opengl
    shader.use();
    GL_CHECK_ERROR(glClearColor(0.0f, 1.0f, 1.0f, 1.0f));
    GL_CHECK_ERROR(glClear(GL_COLOR_BUFFER_BIT));
    GL_CHECK_ERROR(glBindVertexArray(VAO));
    GL_CHECK_ERROR(glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0));

    // read the output into a tensor again
    GL_CHECK_ERROR(glReadBuffer(GL_COLOR_ATTACHMENT1));
    torch::Tensor out = readImageFromOpenGL(eglData.pbufferWidth, eglData.pbufferHeight);

    // destroy the context
    destroyEGL(eglData);

    // return the tensor
    return out;
}
