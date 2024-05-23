#pragma once

#include <torch/extension.h>

// helpers to work with torch
torch::Tensor readImageFromOpenGL(int pbufferWidth, int pbufferHeight);