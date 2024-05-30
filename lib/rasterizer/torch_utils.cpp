#include "torch_utils.h"
#include "gl_utils.h"

#include <GL/gl.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDAGuard.h>

// Function to read the current image from OpenGL into a torch::Tensor
torch::Tensor readImageFromOpenGL(int pbufferWidth, int pbufferHeight)
{
    // Allocate memory to store the pixel data
    std::vector<unsigned char> pixels(pbufferWidth * pbufferHeight * 3); // RGB format
    // Use glReadPixels to read the pixel data from the framebuffer
    GL_CHECK_ERROR(glReadPixels(0, 0, pbufferWidth, pbufferHeight, GL_RGB, GL_UNSIGNED_BYTE, pixels.data()));
    // Copy the pixel data from the vector into a torch::Tensor
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    torch::Tensor out = torch::from_blob(pixels.data(), {pbufferHeight, pbufferWidth, 3}, options).clone();

    return out;
}