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
    int vertexCount;                // number of verticies in the tensor
    int elementCount;               // number of the indices of the vertices
    const float *vertexPtr;         // pointer to vertex tensor
    const uint32_t *elementPtr;     // pointer to vertex tensor
    GLuint glFBO;                   // frame buffer object with multi render targets / deferred rendering
    GLuint glVAO;                   // vertex array object
    GLuint glVBO;                   // vertex buffer object
    GLuint glEBO;                   // element buffer object
    GLuint glOut;                   // output texture that stores barycentric coordinates and triangle idx
    cudaGraphicsResource_t cudaVBO; // vertex buffer object
    cudaGraphicsResource_t cudaEBO; // element buffer object
    cudaGraphicsResource_t cudaOut; // output texture that stores barycentric coordinates and triangle idx
};

struct Fragments
{
    torch::Tensor pix_to_face;
    torch::Tensor bary_coords;
};

Fragments rasterize(torch::Tensor vertices, torch::Tensor indices, int width, int height, int cudaDeviceIdx);
