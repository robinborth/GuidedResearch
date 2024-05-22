#pragma once

#include <torch/extension.h>

torch::Tensor readImageFromOpenGL(int pbufferWidth, int pbufferHeight);
