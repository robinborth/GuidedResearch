#pragma once
#include <GL/gl.h>

GLenum glCheckError_(const char *file, int line);
#define glCheckError() glCheckError_(__FILE__, __LINE__)
void glCheckShaderCompileTimeError(int shader, GLenum shaderType);
void glCheckLinkShaderError(int program);