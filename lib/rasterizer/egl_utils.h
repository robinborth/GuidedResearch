#pragma once
#include <EGL/egl.h>

struct EGLContextData
{
    EGLDisplay eglDpy;
    EGLSurface eglSurf;
    EGLContext eglCtx;
    EGLint numConfigs;
    EGLConfig eglCfg;
    int pbufferWidth = 800;
    int pbufferHeight = 600;
};

void destroyEGL(EGLContextData &eglData);
int initEGL(EGLContextData &eglData);