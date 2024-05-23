#pragma once
#include <EGL/egl.h>

// handling hte context of egl
struct EGLContextData
{
    EGLDisplay eglDpy;
    EGLSurface eglSurf;
    EGLContext eglCtx;
    EGLint numConfigs;
    EGLConfig eglCfg;
    int cudaDeviceIdx = -1;
    int pbufferWidth = 800;
    int pbufferHeight = 600;
};
void destroyEGL(EGLContextData &eglData);
int initEGL(EGLContextData &eglData);
