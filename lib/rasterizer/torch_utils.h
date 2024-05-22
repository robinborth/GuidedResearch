#pragma once

#include <torch/extension.h>

// helpers to work with torch
torch::Tensor readImageFromOpenGL(int pbufferWidth, int pbufferHeight);

// macros to check the correct dtype in torch
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)
