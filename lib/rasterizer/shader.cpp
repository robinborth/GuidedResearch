
#include <GLES3/gl3.h>
#include "shader.h"
#include "utils.h"

Shader::Shader(const char *vShaderCode, const char *fShaderCode)
{
    // compile the vertex shader
    unsigned int vertexShader;
    vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vShaderCode, nullptr);
    glCompileShader(vertexShader);
    glCheckShaderCompileTimeError(vertexShader, GL_VERTEX_SHADER);

    // compile the fragment shader
    unsigned int fragmentShader;
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fShaderCode, nullptr);
    glCompileShader(fragmentShader);
    glCheckShaderCompileTimeError(fragmentShader, GL_FRAGMENT_SHADER);

    // create a shader program that links the shaders together
    ID = glCreateProgram();
    glAttachShader(ID, vertexShader);
    glAttachShader(ID, fragmentShader);
    glLinkProgram(ID);
    glCheckLinkShaderError(ID);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

void Shader::use()
{
    glUseProgram(ID);
}