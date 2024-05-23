#include "gl_utils.h"
#include "gl_context.h"
#include "gl_shader.h"
#include "torch_utils.h"

#include <GLES3/gl3.h>
#include <iostream>

torch::Tensor rasterize(torch::Tensor vertices, torch::Tensor indices, torch::Tensor normals, int width, int height, int cudaDeviceIdx)
{
    // init the context
    EGLContextData eglData;
    eglData.pbufferHeight = height;
    eglData.pbufferWidth = width;
    eglData.cudaDeviceIdx = cudaDeviceIdx;
    initEGL(eglData);

    const char *vShaderCode = "#version 460 core\n" STRINGIFY_SHADER_SOURCE(
        layout(location = 0) in vec4 aPos;
        // layout(location = 1) in vec3 aNormal;

        out VS_OUT {
            vec3 bary;
            // vec3 normal;
        } vs_out;

        void main() {
            gl_Position = aPos;
            vs_out.bary = vec3(0.0, 0.0, 0.0);
            // vs_out.normal = aNormal;
        });

    const char *gShaderCode = "#version 460 core\n" STRINGIFY_SHADER_SOURCE(
        layout(triangles) in;
        layout(triangle_strip, max_vertices = 3) out;

        in VS_OUT {
            vec3 bary;
            // vec3 normal;
        } gs_in[];

        out VS_OUT {
            vec3 bary;
            // vec3 normal;
        } gs_out;

        void main() {
            gs_out.bary = vec3(1.0, 0.0, 0.0);
            // gs_out.normal = gs_in[0].normal;
            gl_Position = gl_in[0].gl_Position;
            EmitVertex();

            gs_out.bary = vec3(0.0, 1.0, 0.0);
            // gs_out.normal = gs_in[1].normal;
            gl_Position = gl_in[1].gl_Position;
            EmitVertex();

            gs_out.bary = vec3(0.0, 0.0, 1.0);
            // gs_out.normal = gs_in[2].normal;
            gl_Position = gl_in[2].gl_Position;
            EmitVertex();

            EndPrimitive();
        });

    const char *fShaderCode = "#version 460 core\n" STRINGIFY_SHADER_SOURCE(
        layout(location = 0) out vec3 gBary;
        // layout(location = 1) out int gPrimitiveID;
        // layout(location = 2) out vec3 gNormal;

        in VS_OUT {
            vec3 bary;
            // vec3 normal;
        } fs_in;

        void main() {
            gBary = fs_in.bary;
            // gBary = vec4(fs_in.bary, 1.0);
            // gPrimitiveID = gl_PrimitiveID;
            // gNormal = fs_in.normal;
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

    // GL_CHECK_ERROR(glBufferData(GL_ARRAY_BUFFER, normals.numel() * sizeof(float), normals.data_ptr<float>(), GL_STATIC_DRAW));
    // GL_CHECK_ERROR(glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void *)4));

    GL_CHECK_ERROR(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO));
    GL_CHECK_ERROR(glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.numel() * sizeof(uint32_t), indices.data_ptr<uint32_t>(), GL_STATIC_DRAW));

    GL_CHECK_ERROR(glEnableVertexAttribArray(0));
    GL_CHECK_ERROR(glBindVertexArray(0));

    unsigned int gBuffer;
    GL_CHECK_ERROR(glGenFramebuffers(1, &gBuffer));
    GL_CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, gBuffer));
    unsigned int gBary, gPrimitiveID, gNormal;

    // bary centric coordinates
    GL_CHECK_ERROR(glGenTextures(1, &gBary));
    GL_CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, gBary));
    GL_CHECK_ERROR(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, eglData.pbufferWidth, eglData.pbufferHeight, 0, GL_RGB, GL_FLOAT, nullptr));
    GL_CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
    GL_CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
    GL_CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, 0));
    GL_CHECK_ERROR(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gBary, 0));

    // triangle ids
    // GL_CHECK_ERROR(glGenTextures(1, &gPrimitiveID));
    // GL_CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, gPrimitiveID));
    // GL_CHECK_ERROR(glTexImage2D(GL_TEXTURE_2D, 0, GL_INT, eglData.pbufferWidth, eglData.pbufferHeight, 0, GL_INT, GL_INT, nullptr));
    // GL_CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
    // GL_CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
    // GL_CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, 0));
    // GL_CHECK_ERROR(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, gPrimitiveID, 0));

    // normal maps
    // GL_CHECK_ERROR(glGenTextures(1, &gNormal));
    // GL_CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, gNormal));
    // GL_CHECK_ERROR(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, eglData.pbufferWidth, eglData.pbufferHeight, 0, GL_RGB, GL_FLOAT, nullptr));
    // GL_CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
    // GL_CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
    // GL_CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, 0));
    // GL_CHECK_ERROR(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, gNormal, 0));

    // unsigned int attachments[3] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2};
    // GL_CHECK_ERROR(glDrawBuffers(3, attachments));

    unsigned int attachments[1] = {GL_COLOR_ATTACHMENT0};
    GL_CHECK_ERROR(glDrawBuffers(1, attachments));

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
    GL_CHECK_ERROR(glReadBuffer(GL_COLOR_ATTACHMENT0));
    torch::Tensor out = readImageFromOpenGL(eglData.pbufferWidth, eglData.pbufferHeight);

    // destroy the context
    destroyEGL(eglData);

    // return the tensor
    return out;
}
