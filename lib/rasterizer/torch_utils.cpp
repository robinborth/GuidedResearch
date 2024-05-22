#include "torch_utils.h"
#include <GL/gl.h>

// Function to read the current image from OpenGL into a torch::Tensor
torch::Tensor readImageFromOpenGL(int pbufferWidth, int pbufferHeight)
{
    // Allocate memory to store the pixel data
    std::vector<unsigned char> pixels(pbufferWidth * pbufferHeight * 4); // RGBA format

    // Use glReadPixels to read the pixel data from the framebuffer
    glReadPixels(0, 0, pbufferWidth, pbufferHeight, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());

    // Copy the pixel data from the vector into a torch::Tensor
    // Note: Assuming RGBA format
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    return torch::from_blob(pixels.data(), {pbufferHeight, pbufferWidth, 4}, options).clone();
}