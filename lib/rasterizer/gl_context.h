#pragma once
#include <EGL/egl.h>

class GLContext
{
  public:
    EGLDisplay display;
    EGLContext context;
    int width;
    int height;
    int cudaDeviceIdx;

    GLContext(int width, int height, int cudaDeviceIdx);
    void destroy();

  private:
    EGLDisplay getCudaDisplay(int cudaDeviceIdx);
};
