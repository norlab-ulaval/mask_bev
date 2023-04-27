#version 330 core

layout (location = 0) in vec3 vertPosition;

out vec3 fragPos;

uniform mat4 vertProjMat;
uniform mat4 vertViewMat;
uniform mat4 vertModelMat;
uniform float vertPointSize;
uniform float vertRenderMode;

void main()
{
    vec3 pos = vertPosition.xyz;

    gl_Position = vertProjMat * vertViewMat * vertModelMat * vec4(pos, 1.0);

    vec4 vertPos4 = vertViewMat * vec4(pos, 1.0);

    fragPos = vec3(vertPos4) / vertPos4.w;
}
