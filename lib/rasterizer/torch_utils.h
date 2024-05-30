#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

// helpers to work with torch
torch::Tensor readImageFromOpenGL(int pbufferWidth, int pbufferHeight);

// Inline function for CUDA error checking
inline void checkCudaError(cudaError_t err, const char *file, int line, const char *call)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error at " << file << ":" << line << " - "
                  << cudaGetErrorString(err) << " (" << err << ") [" << call << "]" << std::endl;
        TORCH_CHECK(false, "CUDA error at ", file, ":", line, " - ",
                    cudaGetErrorString(err), " (", err, ") [", call, "]");
    }
}

// Macro to wrap the inline function for ease of use
#define CUDA_CHECK_ERROR(CUDA_CALL) checkCudaError((CUDA_CALL), __FILE__, __LINE__, #CUDA_CALL)