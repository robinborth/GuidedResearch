#pragma once
#include <EGL/egl.h>

struct GLContext
{
    EGLDisplay display;
    EGLContext context;
    int width;
    int height;
    int cudaDeviceIdx;
};

void destroyContext(GLContext &glctx);
int initContext(GLContext &glctx);
