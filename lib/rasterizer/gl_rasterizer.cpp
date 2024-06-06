#include "gl_rasterizer.h"
#include "gl_utils.h"
#include "gl_context.h"
#include "gl_shader.h"
#include "torch_utils.h"

#include <GLES3/gl3.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_gl_interop.h>
#include <iostream>

Fragments rasterize(GLContext glctx, torch::Tensor vertices, torch::Tensor indices)
{
    // define the rasterization state
    RasterizeGLState s;
    s.batchSize = vertices.size(0);
    s.vertexPerElement = vertices.size(1);
    s.vertexCount = vertices.numel();
    s.vertexPtr = vertices.data_ptr<float>();
    s.elementCount = indices.numel();
    s.elementPtr = indices.data_ptr<uint32_t>();
    GL_CHECK_ERROR(glGenFramebuffers(1, &s.glFBO));
    GL_CHECK_ERROR(glGenVertexArrays(1, &s.glVAO));
    GL_CHECK_ERROR(glGenBuffers(1, &s.glVBO));
    GL_CHECK_ERROR(glGenBuffers(1, &s.glEBO));
    GL_CHECK_ERROR(glGenTextures(1, &s.glOut));

    // access the current cuda stream that is used in pytorch
    const at::cuda::OptionalCUDAGuard device_guard(device_of(vertices));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // initilize the fragment shader
    FragmentShader shader = FragmentShader();

    // enable the vertex attribute object
    GL_CHECK_ERROR(glBindVertexArray(s.glVAO));
    // // vertex data is on CPU -> allocate memory on the GPU
    // GL_CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, s.glVBO));
    // GL_CHECK_ERROR(glBufferData(GL_ARRAY_BUFFER, s.vertexCount * sizeof(float), s.vertexPtr, GL_STATIC_DRAW));
    // GL_CHECK_ERROR(glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0));
    // vertex data is on GPU -> copy data to the GPU
    GL_CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, s.glVBO));
    void *glVertexPtr = nullptr;
    size_t vertexBytes = 0;
    GL_CHECK_ERROR(glBufferData(GL_ARRAY_BUFFER, s.vertexCount * sizeof(float), nullptr, GL_DYNAMIC_DRAW));
    GL_CHECK_ERROR(glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0));
    CUDA_CHECK_ERROR(cudaGraphicsGLRegisterBuffer(&s.cudaVBO, s.glVBO, cudaGraphicsRegisterFlagsWriteDiscard));
    CUDA_CHECK_ERROR(cudaGraphicsMapResources(1, &s.cudaVBO, stream));
    CUDA_CHECK_ERROR(cudaGraphicsResourceGetMappedPointer(&glVertexPtr, &vertexBytes, s.cudaVBO));
    CUDA_CHECK_ERROR(cudaMemcpyAsync(glVertexPtr, s.vertexPtr, s.vertexCount * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK_ERROR(cudaGraphicsUnmapResources(1, &s.cudaVBO, stream));
    GL_CHECK_ERROR(glEnableVertexAttribArray(0));
    // // element data is on CPU -> allocate memory on the GPU
    // GL_CHECK_ERROR(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s.glEBO));
    // GL_CHECK_ERROR(glBufferData(GL_ELEMENT_ARRAY_BUFFER, s.elementCount * sizeof(uint32_t), s.elementPtr, GL_STATIC_DRAW));
    // element data is on CPU -> allocate memory on the GPU
    GL_CHECK_ERROR(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s.glEBO));
    void *glElementPtr = nullptr;
    size_t elementBytes = 0;
    GL_CHECK_ERROR(glBufferData(GL_ELEMENT_ARRAY_BUFFER, s.elementCount * sizeof(uint32_t), nullptr, GL_DYNAMIC_DRAW));
    CUDA_CHECK_ERROR(cudaGraphicsGLRegisterBuffer(&s.cudaEBO, s.glEBO, cudaGraphicsRegisterFlagsWriteDiscard));
    CUDA_CHECK_ERROR(cudaGraphicsMapResources(1, &s.cudaEBO, stream));
    CUDA_CHECK_ERROR(cudaGraphicsResourceGetMappedPointer(&glElementPtr, &elementBytes, s.cudaEBO));
    CUDA_CHECK_ERROR(cudaMemcpyAsync(glElementPtr, s.elementPtr, s.elementCount * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK_ERROR(cudaGraphicsUnmapResources(1, &s.cudaEBO, stream));
    // unbind the vertex array
    GL_CHECK_ERROR(glBindVertexArray(0));

    // bind the framebuffer
    GL_CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, s.glFBO));
    GL_CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, s.glOut));
    GL_CHECK_ERROR(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, glctx.width, glctx.height, 0, GL_RGBA, GL_FLOAT, nullptr));
    GL_CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
    GL_CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
    GL_CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, 0));
    GL_CHECK_ERROR(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, s.glOut, 0));

    unsigned int rbo;
    GL_CHECK_ERROR(glGenRenderbuffers(1, &rbo));
    GL_CHECK_ERROR(glBindRenderbuffer(GL_RENDERBUFFER, rbo));
    GL_CHECK_ERROR(glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, glctx.width, glctx.height));
    GL_CHECK_ERROR(glBindRenderbuffer(GL_RENDERBUFFER, 0));
    GL_CHECK_ERROR(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo));

    // add attachments to the framebuffer
    unsigned int attachments[1] = {GL_COLOR_ATTACHMENT0};
    GL_CHECK_ERROR(glDrawBuffers(1, attachments));
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "ERROR FRAMEBUFFER NOT COMPLETE" << std::endl;

    // unbind the framebuffer
    GL_CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, 0));

    // rasterizes the vertices using opengl
    shader.use();
    GL_CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, s.glFBO));
    GL_CHECK_ERROR(glViewport(0, 0, glctx.width, glctx.height));
    GL_CHECK_ERROR(glClearColor(0.0f, 0.0f, 0.0f, -1.0f));
    GL_CHECK_ERROR(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
    GL_CHECK_ERROR(glEnable(GL_DEPTH_TEST));
    GL_CHECK_ERROR(glDepthFunc(GL_LESS));
    GL_CHECK_ERROR(glBindVertexArray(s.glVAO));
    GL_CHECK_ERROR(glDrawElements(GL_TRIANGLES, s.elementCount, GL_UNSIGNED_INT, 0));

    // allocate output tensors.
    torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor out = torch::empty({glctx.height, glctx.width, 4}, opts);
    float *outputPtr = out.data_ptr<float>();
    // register to cuda
    cudaArray_t cudaOut = 0;
    CUDA_CHECK_ERROR(cudaGraphicsGLRegisterImage(&s.cudaOut, s.glOut, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
    CUDA_CHECK_ERROR(cudaGraphicsMapResources(1, &s.cudaOut, stream));
    CUDA_CHECK_ERROR(cudaGraphicsSubResourceGetMappedArray(&cudaOut, s.cudaOut, 0, 0));
    // perform the asynchronous copy
    CUDA_CHECK_ERROR(cudaMemcpy2DFromArrayAsync(
        outputPtr,                       // Destination pointer
        glctx.width * 4 * sizeof(float), // Destination pitch
        cudaOut,                         // Source array
        0, 0,                            // Offset in the source array
        glctx.width * 4 * sizeof(float), // Width of the 2D region to copy in bytes
        glctx.height,                    // Height of the 2D region to copy in rows
        cudaMemcpyDeviceToDevice,        // Copy kind
        stream));
    CUDA_CHECK_ERROR(cudaGraphicsUnmapResources(1, &s.cudaOut, stream));
    CUDA_CHECK_ERROR(cudaGraphicsUnregisterResource(s.cudaOut));

    Fragments fragments;
    fragments.bary_coords = out.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 3)}).clone();
    fragments.pix_to_face = out.index({torch::indexing::Slice(), torch::indexing::Slice(), 3}).to(torch::kInt32).clone();

    // unregister the context, we allready unmapped them
    CUDA_CHECK_ERROR(cudaGraphicsUnregisterResource(s.cudaVBO));
    CUDA_CHECK_ERROR(cudaGraphicsUnregisterResource(s.cudaEBO));

    // delete the buffers and textures
    GL_CHECK_ERROR(glDeleteFramebuffers(1, &s.glFBO));
    GL_CHECK_ERROR(glDeleteVertexArrays(1, &s.glVAO));
    GL_CHECK_ERROR(glDeleteBuffers(1, &s.glVBO));
    GL_CHECK_ERROR(glDeleteBuffers(1, &s.glEBO));
    GL_CHECK_ERROR(glDeleteTextures(1, &s.glOut));
    GL_CHECK_ERROR(glDeleteRenderbuffers(1, &rbo));

    // destroy the context
    glctx.destroy();

    // return the tensor
    return fragments;
}