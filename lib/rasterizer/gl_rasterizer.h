#pragma once

#include <torch/extension.h>
#include <GL/gl.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
// #include <cuda_gl_interop.h>
#include "gl_context.h"
#include "gl_shader.h"

struct RasterizeGLState
{
    int vertexCount;                    // number of verticies in the tensor
    int elementCount;                   // number of the indices of the vertices
    const float *vertexPtr;          // pointer to vertex tensor
    const uint32_t *elementPtr;          // pointer to vertex tensor
    GLuint glFBO;                       // frame buffer object with multi render targets / deferred rendering
    GLuint glVAO;                       // vertex array object
    GLuint glVBO;                       // vertex buffer object
    GLuint glEBO;                       // element buffer object
    GLuint glOutBary;                   // output texture that stores barycentric coordinates
    cudaGraphicsResource_t cudaVBO;     // vertex buffer object
    cudaGraphicsResource_t cudaEBO;     // element buffer object
    cudaGraphicsResource_t cudaOutBary; // output texture that stores barycentric coordinates
};

torch::Tensor rasterize(torch::Tensor vertices, torch::Tensor indices, int width, int height, int cudaDeviceIdx);
Shader initShader();

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