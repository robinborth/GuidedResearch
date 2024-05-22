#include "gl_utils.h"

#include <GLES3/gl3.h>
#include <iostream>

constexpr const char *shaderTypeToString(GLenum shaderType)
{
    if (shaderType == GL_VERTEX_SHADER)
        return "VERTEX";
    if (shaderType == GL_FRAGMENT_SHADER)
        return "FRAGMENT";
    return "UNKNOWN";
}

void glCheckShaderCompileTimeError(int shader, GLenum shaderType)
{
    int success;
    char infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

    if (!success)
    {
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::" << shaderTypeToString(shaderType) << "::COMPILATION_FAILED\n"
                  << infoLog << std::endl;
    }
}

void glCheckLinkShaderError(int program)
{
    int success;
    char infoLog[512];
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success)
    {
        glGetProgramInfoLog(program, 512, NULL, infoLog);
        std::cout << "ERROR::PROGRAM::COMPILATION_FAILED\n"
                  << infoLog << std::endl;
    }
}

GLenum glCheckError(const char *file, int line)
{
    GLenum errorCode;
    while ((errorCode = glGetError()) != GL_NO_ERROR)
    {
        std::string error;
        switch (errorCode)
        {
        case GL_INVALID_ENUM:
            error = "INVALID_ENUM";
            break;
        case GL_INVALID_VALUE:
            error = "INVALID_VALUE";
            break;
        case GL_INVALID_OPERATION:
            error = "INVALID_OPERATION";
            break;
        case GL_STACK_OVERFLOW:
            error = "GL_STACK_OVERFLOW";
            break;
        case GL_STACK_UNDERFLOW:
            error = "GL_STACK_UNDERFLOW";
            break;
        case GL_OUT_OF_MEMORY:
            error = "OUT_OF_MEMORY";
            break;
        case GL_INVALID_FRAMEBUFFER_OPERATION:
            error = "INVALID_FRAMEBUFFER_OPERATION";
            break;
        case GL_TABLE_TOO_LARGE:
            error = "GL_TABLE_TOO_LARGE";
            break;
        case GL_CONTEXT_LOST:
            error = "GL_CONTEXT_LOST";
            break;
        }
        std::cout << error << " | " << file << " (" << line << ")" << std::endl;
    }
    return errorCode;
}