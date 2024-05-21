#pragma once
#include <GL/gl.h>

#define glCheckError() glCheckError_(__FILE__, __LINE__)
void glCheckShaderCompileTimeError(int shader, GLenum shaderType);
void glCheckLinkShaderError(int program);