#pragma once

#include <GL/gl.h>
#include <vector>
#include "gl_rasterizer.h"

#define STRINGIFY_SHADER_SOURCE(x) #x

class Shader
{
  public:
    unsigned int ID;
    Shader(const char *vShaderCode, const char *fShaderCode);
    Shader(const char *vShaderCode, const char *gShaderCode, const char *fShaderCode);
    void use();

  private:
    unsigned int vertexShader;
    unsigned int fragmentShader;
    unsigned int geometryShader;

    void compileShader(unsigned int &shader, const char *shaderCode, GLenum shader_type);

    template <typename... T>
    void linkProgram(T... shaders);
    void linkProgram(const std::vector<unsigned int> &shaders);
};

class FragmentShader : public Shader
{
  public:
    FragmentShader() : Shader(vShaderCode(), gShaderCode(), fShaderCode()){};

  private:
    static const char *vShaderCode();
    static const char *gShaderCode();
    static const char *fShaderCode();
};