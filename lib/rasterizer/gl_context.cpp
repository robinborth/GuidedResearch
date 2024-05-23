#include "gl_utils.h"
#include "gl_context.h"

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <GL/gl.h>
#include <iostream>

static EGLDisplay getCudaDisplay(int cudaDeviceIdx)
{
    typedef EGLBoolean (*eglQueryDevicesEXT_t)(EGLint, EGLDeviceEXT, EGLint *);
    typedef EGLBoolean (*eglQueryDeviceAttribEXT_t)(EGLDeviceEXT, EGLint, EGLAttrib *);
    typedef EGLDisplay (*eglGetPlatformDisplayEXT_t)(EGLenum, void *, const EGLint *);

    eglQueryDevicesEXT_t eglQueryDevicesEXT = (eglQueryDevicesEXT_t)eglGetProcAddress("eglQueryDevicesEXT");
    if (!eglQueryDevicesEXT)
    {
        std::cout << "eglGetProcAddress(\"eglQueryDevicesEXT\") failed" << std::endl;
        return 0;
    }

    eglQueryDeviceAttribEXT_t eglQueryDeviceAttribEXT = (eglQueryDeviceAttribEXT_t)eglGetProcAddress("eglQueryDeviceAttribEXT");
    if (!eglQueryDeviceAttribEXT)
    {
        std::cout << "eglGetProcAddress(\"eglQueryDeviceAttribEXT\") failed" << std::endl;
        return 0;
    }

    eglGetPlatformDisplayEXT_t eglGetPlatformDisplayEXT = (eglGetPlatformDisplayEXT_t)eglGetProcAddress("eglGetPlatformDisplayEXT");
    if (!eglGetPlatformDisplayEXT)
    {
        std::cout << "eglGetProcAddress(\"eglGetPlatformDisplayEXT\") failed" << std::endl;
        return 0;
    }

    int num_devices = 0;
    eglQueryDevicesEXT(0, 0, &num_devices);
    if (!num_devices)
    {
        std::cout << "No EGL devices found" << std::endl;
        return 0;
    }

    EGLDisplay display = 0;
    EGLDeviceEXT *devices = (EGLDeviceEXT *)malloc(num_devices * sizeof(void *));
    eglQueryDevicesEXT(num_devices, devices, &num_devices);
    for (int i = 0; i < num_devices; i++)
    {
        EGLDeviceEXT device = devices[i];
        intptr_t value = -1;
        if (eglQueryDeviceAttribEXT(device, EGL_CUDA_DEVICE_NV, &value) && value == cudaDeviceIdx)
        {
            display = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, device, 0);
            break;
        }
    }

    free(devices);
    return display;
}

int initEGL(EGLContextData &eglData)
{
    // 1. create the display either from the current cuda device or use the default egl display
    eglData.eglDpy = 0;
    if (eglData.cudaDeviceIdx >= 0)
    {
        std::cout << "Creating GL context for cuda device " << eglData.cudaDeviceIdx << std::endl;
        eglData.eglDpy = getCudaDisplay(eglData.cudaDeviceIdx);
        if (!eglData.eglDpy)
        {

            EGLint error = eglGetError();
            std::cout << "Failed, falling back to default display" << error << std::endl;
        }
    }
    if (!eglData.eglDpy)
    {
        std::cout << "Creating GL context for default device " << std::endl;
        eglData.eglDpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if (eglData.eglDpy == EGL_NO_DISPLAY)
        {
            EGLint error = eglGetError();
            std::cout << "Unable to get EGL display: " << error << std::endl;
            return -1;
        }
    }

    // 2. initilize egl
    EGLint major, minor;
    EGLBoolean success = eglInitialize(eglData.eglDpy, &major, &minor);
    if (!success)
    {
        EGLint error = eglGetError();
        std::cout << "Unable to initialize EGL. EGL error: " << error << std::endl;
        return -1;
    }
    std::cout << "Successfully initialized EGL version " << major << "." << minor << std::endl;

    // 3. choose the config
    static const EGLint configAttribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_BLUE_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_RED_SIZE, 8,
        EGL_DEPTH_SIZE, 8,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_NONE};
    eglChooseConfig(eglData.eglDpy, configAttribs, &eglData.eglCfg, 1, &eglData.numConfigs);

    // 4. create a surface
    static const EGLint pbufferAttribs[] = {
        EGL_WIDTH,
        eglData.pbufferWidth,
        EGL_HEIGHT,
        eglData.pbufferHeight,
        EGL_NONE,
    };
    eglData.eglSurf = eglCreatePbufferSurface(eglData.eglDpy, eglData.eglCfg, pbufferAttribs);

    // 5. bind the api
    eglBindAPI(EGL_OPENGL_API);

    // 6. create the context
    eglData.eglCtx = eglCreateContext(eglData.eglDpy, eglData.eglCfg, EGL_NO_CONTEXT, nullptr);
    eglMakeCurrent(eglData.eglDpy, eglData.eglSurf, eglData.eglSurf, eglData.eglCtx);

    // 7. check that the OpenGL version is correct
    const unsigned char *version = glGetString(GL_VERSION);
    if (!version)
    {
        std::cout << "Unable to retrieve OpenGL version." << std::endl;
        return -1;
    }
    std::cout << "Successfully initialized OpenGL version " << version << std::endl;

    return 0;
}

void destroyEGL(EGLContextData &eglData)
{
    eglTerminate(eglData.eglDpy);
}
