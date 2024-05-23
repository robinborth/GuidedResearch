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

Shader initShader()
{
    const char *vShaderCode = "#version 460 core\n" STRINGIFY_SHADER_SOURCE(
        layout(location = 0) in vec4 aPos;

        out VS_OUT {
            vec3 bary;
        } vs_out;

        void main() {
            gl_Position = aPos;
            vs_out.bary = vec3(0.0, 0.0, 0.0);
        });

    const char *gShaderCode = "#version 460 core\n" STRINGIFY_SHADER_SOURCE(
        layout(triangles) in;
        layout(triangle_strip, max_vertices = 3) out;

        in VS_OUT {
            vec3 bary;
        } gs_in[];

        out VS_OUT {
            vec3 bary;
        } gs_out;

        void main() {
            gs_out.bary = vec3(1.0, 0.0, 0.0);
            gl_Position = gl_in[0].gl_Position;
            EmitVertex();
            gs_out.bary = vec3(0.0, 1.0, 0.0);
            gl_Position = gl_in[1].gl_Position;
            EmitVertex();
            gs_out.bary = vec3(0.0, 0.0, 1.0);
            gl_Position = gl_in[2].gl_Position;
            EmitVertex();
            EndPrimitive();
        });

    const char *fShaderCode = "#version 460 core\n" STRINGIFY_SHADER_SOURCE(
        layout(location = 0) out vec3 gBary;
        // layout(location = 1) out int gPrimitiveID;

        in VS_OUT {
            vec3 bary;
        } fs_in;

        void main() {
            gBary = fs_in.bary;
            // gPrimitiveID = gl_PrimitiveID;
        });
    Shader shader(vShaderCode, gShaderCode, fShaderCode);
    return shader;
}

torch::Tensor rasterize(torch::Tensor vertices, torch::Tensor indices, int width, int height, int cudaDeviceIdx)
{
    GLContext glctx;
    glctx.width = width;
    glctx.height = width;
    glctx.cudaDeviceIdx = cudaDeviceIdx;
    initContext(glctx);

    RasterizeGLState s;
    s.vertexCount = vertices.numel();
    s.elementCount = indices.numel();
    GL_CHECK_ERROR(glGenFramebuffers(1, &s.glFBO));
    GL_CHECK_ERROR(glGenVertexArrays(1, &s.glVAO));
    GL_CHECK_ERROR(glGenBuffers(1, &s.glVBO));
    GL_CHECK_ERROR(glGenBuffers(1, &s.glEBO));
    GL_CHECK_ERROR(glGenTextures(1, &s.glOutBary));

    // access the current cuda stream that is used in pytorch
    // const at::cuda::OptionalCUDAGuard device_guard(device_of(vertices));
    // cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    Shader shader = initShader();

    GL_CHECK_ERROR(glBindVertexArray(s.glVAO));
    GL_CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, s.glVBO));
    GL_CHECK_ERROR(glBufferData(GL_ARRAY_BUFFER, s.vertexCount * sizeof(float), vertices.data_ptr<float>(), GL_STATIC_DRAW));
    GL_CHECK_ERROR(glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0));

    // cudaGraphicsResource_t cudaVertBuffer;
    // GL_CHECK_ERROR(glBufferData(GL_ARRAY_BUFFER, vertices.numel() * sizeof(float), nullptr, GL_STATIC_DRAW));
    // GL_CHECK_ERROR(glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0));
    // cudaGraphicsGLRegisterBuffer(&cudaVertBuffer, VBO, cudaGraphicsRegisterFlagsWriteDiscard);

    GL_CHECK_ERROR(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s.glEBO));
    GL_CHECK_ERROR(glBufferData(GL_ELEMENT_ARRAY_BUFFER, s.elementCount * sizeof(uint32_t), indices.data_ptr<uint32_t>(), GL_STATIC_DRAW));

    GL_CHECK_ERROR(glEnableVertexAttribArray(0));
    GL_CHECK_ERROR(glBindVertexArray(0));

    // const float *vertPtr = vertices.data_ptr<float>();
    // void *glVertPtr = nullptr;
    // size_t vertBytes = 0;
    // cudaGraphicsMapResources(1, &cudaVertBuffer, stream);
    // cudaGraphicsResourceGetMappedPointer(&glVertPtr, &vertBytes, cudaVertBuffer);
    // cudaMemcpyAsync(glVertPtr, vertPtr, vertices.size(0) * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    // cudaGraphicsUnmapResources(1, &cudaVertBuffer, stream);

    // cudaGraphicsResource_t cudaIdxBuffer;
    // const uint32_t *idxPtr = indices.data_ptr<uint32_t>();
    // void *glIdxPtr = nullptr;
    // size_t idxBytes = 0;
    // cudaGraphicsResourceGetMappedPointer(&glIdxPtr, &idxBytes, cudaIdxBuffer);
    // cudaMemcpyAsync(glIdxPtr, idxPtr, indices.size(0) * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream);

    // Bind the framebuffer
    GL_CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, s.glFBO));

    GL_CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, s.glOutBary));
    GL_CHECK_ERROR(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, glctx.width, glctx.height, 0, GL_RGB, GL_FLOAT, nullptr));
    GL_CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
    GL_CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
    GL_CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, 0));
    GL_CHECK_ERROR(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, s.glOutBary, 0));

    unsigned int attachments[1] = {GL_COLOR_ATTACHMENT0};
    GL_CHECK_ERROR(glDrawBuffers(1, attachments));
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "ERROR FRAMEBUFFER NOT COMPLETE" << std::endl;

    // rasterizes the vertices using opengl
    shader.use();
    GL_CHECK_ERROR(glClearColor(0.0f, 0.0f, 0.0f, 1.0f));
    GL_CHECK_ERROR(glClear(GL_COLOR_BUFFER_BIT));
    GL_CHECK_ERROR(glBindVertexArray(s.glVAO));
    GL_CHECK_ERROR(glDrawElements(GL_TRIANGLES, s.elementCount, GL_UNSIGNED_INT, 0));

    // read the output into a tensor again
    GL_CHECK_ERROR(glReadBuffer(GL_COLOR_ATTACHMENT0));
    torch::Tensor out = readImageFromOpenGL(glctx.width, glctx.height);

    // Allocate output tensors.
    // torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    // torch::Tensor out = torch::empty({height, width, 3}, opts);
    // float *outputPtr = out.data_ptr<float>();

    // Copy rasterized results into CUDA buffers.
    // rasterizeCopyResults(, stream, outputPtr, width, height);

    // destroy the context
    destroyContext(glctx);

    // return the tensor
    return out;
}

// void rasterizeCopyResults(cudaGraphicsResource_t *cudaColorBuffer, cudaStream_t stream, float *outputPtr, int width, int height)
// {
//     // Copy color buffers to output tensors.
//     cudaArray_t array = 0;
//     cudaGraphicsMapResources(1, cudaColorBuffer, stream);
//     cudaGraphicsSubResourceGetMappedArray(&array, *cudaColorBuffer, 0, 0);
//     cudaMemcpy3DParms p = {0};
//     p.srcArray = array;
//     p.dstPtr.ptr = outputPtr;
//     p.dstPtr.pitch = width * 4 * sizeof(float);
//     p.dstPtr.xsize = width;
//     p.dstPtr.ysize = height;
//     p.extent.width = width;
//     p.extent.height = height;
//     p.kind = cudaMemcpyDeviceToDevice;
//     cudaMemcpy3DAsync(&p, stream);
//     cudaGraphicsUnmapResources(1, cudaColorBuffer, stream);
// }