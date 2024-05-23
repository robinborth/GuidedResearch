#pragma once

#include <GL/gl.h>

// error checking and debug utils
GLenum glCheckError(const char *file, int line);
void glCheckShaderCompileTimeError(int shader, GLenum shaderType);
void glCheckLinkShaderError(int program);

// macros for checking opengl errors
#define GL_CHECK_ERROR(GL_CALL) \
    GL_CALL;                    \
    glCheckError(__FILE__, __LINE__);
