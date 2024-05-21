#include "egl_utils.h"
#include <iostream>

int initEGL(EGLContextData &eglData)
{
    // 1. create the display
    EGLDisplay eglDpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (eglDpy == EGL_NO_DISPLAY)
    {
        std::cout << "Unable to get EGL display." << std::endl;
        return -1;
    }

    // 2. initilize egl
    EGLint major, minor;
    EGLBoolean success = eglInitialize(eglDpy, &major, &minor);
    if (!success)
    {
        EGLint error = eglGetError();
        std::cout << "Unable to initialize EGL. EGL error: " << error << std::endl;
        return -1;
    }
    std::cout << "Successfully initialized EGL version " << major << "." << minor << std::endl;

    // 3. choose the config
    EGLint numConfigs;
    EGLConfig eglCfg;
    EGLint configAttribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_BLUE_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_RED_SIZE, 8,
        EGL_DEPTH_SIZE, 8,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_NONE};
    eglChooseConfig(eglData.eglDpy, configAttribs, &eglCfg, 1, &numConfigs);

    // 4. create a surface
    EGLint pbufferAttribs[] = {
        EGL_WIDTH,
        eglData.pbufferWidth,
        EGL_HEIGHT,
        eglData.pbufferHeight,
        EGL_NONE,
    };
    eglData.eglSurf = eglCreatePbufferSurface(eglData.eglDpy, eglCfg, pbufferAttribs);

    // 5. bind the api
    eglBindAPI(EGL_OPENGL_API);

    // 6. create the context
    eglData.eglCtx = eglCreateContext(eglData.eglDpy, eglCfg, EGL_NO_CONTEXT, nullptr);
    eglMakeCurrent(eglData.eglDpy, eglData.eglSurf, eglData.eglSurf, eglData.eglCtx);

    // 7. print the success of the inilization
    std::cout << "EGL initialized successfully" << std::endl;

    return 0;
}

void destroyEGL(EGLContextData &eglData)
{
    eglTerminate(eglData.eglDpy);
}
