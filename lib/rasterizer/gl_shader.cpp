
#include <GLES3/gl3.h>

#include "gl_shader.h"
#include "gl_utils.h"

Shader::Shader(const char *vShaderCode, const char *gShaderCode, const char *fShaderCode)
{
    compileShader(vertexShader, vShaderCode, GL_VERTEX_SHADER);
    compileShader(geometryShader, gShaderCode, GL_GEOMETRY_SHADER);
    compileShader(fragmentShader, fShaderCode, GL_FRAGMENT_SHADER);
    linkProgram(vertexShader, geometryShader, fragmentShader);
}

Shader::Shader(const char *vShaderCode, const char *fShaderCode)
{
    compileShader(vertexShader, vShaderCode, GL_VERTEX_SHADER);
    compileShader(fragmentShader, fShaderCode, GL_FRAGMENT_SHADER);
    linkProgram(vertexShader, fragmentShader);
}

void Shader::use()
{
    glUseProgram(ID);
}

void Shader::compileShader(unsigned int &shader, const char *shaderCode, GLenum shader_type)
{
    shader = GL_CHECK_ERROR(glCreateShader(shader_type));
    GL_CHECK_ERROR(glShaderSource(shader, 1, &shaderCode, nullptr));
    GL_CHECK_ERROR(glCompileShader(shader));
    glCheckShaderCompileTimeError(shader, shader_type);
}

template <typename... T>
void Shader::linkProgram(T... shaders)
{
    std::vector<unsigned int> shaderVec = {shaders...};
    linkProgram(shaderVec);
}

void Shader::linkProgram(const std::vector<unsigned int> &shaders)
{
    ID = GL_CHECK_ERROR(glCreateProgram());
    for (auto shader : shaders)
        GL_CHECK_ERROR(glAttachShader(ID, shader));
    GL_CHECK_ERROR(glLinkProgram(ID));
    glCheckLinkShaderError(ID);
    for (auto shader : shaders)
        GL_CHECK_ERROR(glDeleteShader(shader));
}

// ----------------------------------------------------------------------------------------

const char *FragmentShader::vShaderCode()
{
    return "#version 460 core\n" STRINGIFY_SHADER_SOURCE(
        layout(location = 0) in vec4 aPos;

        out VS_OUT {
            vec3 bary;
        } vs_out;

        void main() {
            gl_Position = aPos;
            vs_out.bary = vec3(0.0, 0.0, 0.0);
        });
}

const char *FragmentShader::gShaderCode()
{
    return "#version 460 core\n" STRINGIFY_SHADER_SOURCE(
        layout(triangles) in;
        layout(triangle_strip, max_vertices = 3) out;

        in VS_OUT {
            vec3 bary;
        } gs_in[];

        out VS_OUT {
            vec3 bary;
        } gs_out;

        void main() {
            gs_out.bary = vec3(1.0, 0.0, 0.0);
            gl_Position = gl_in[0].gl_Position;
            EmitVertex();
            gs_out.bary = vec3(0.0, 1.0, 0.0);
            gl_Position = gl_in[1].gl_Position;
            EmitVertex();
            gs_out.bary = vec3(0.0, 0.0, 1.0);
            gl_Position = gl_in[2].gl_Position;
            EmitVertex();
            EndPrimitive();
        });
}

const char *FragmentShader::fShaderCode()
{
    return "#version 460 core\n" STRINGIFY_SHADER_SOURCE(
        layout(location = 0) out vec4 gOut;

        in VS_OUT {
            vec3 bary;
        } fs_in;

        void main() {
            float primitiveID = float(gl_PrimitiveID);
            gOut = vec4(fs_in.bary, primitiveID);
        });
};