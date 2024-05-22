#pragma once

#define STRINGIFY_SHADER_SOURCE(x) #x

class Shader
{
  public:
    unsigned int ID;
    Shader(const char *vShaderCode, const char *fShaderCode);
    void use();
};
