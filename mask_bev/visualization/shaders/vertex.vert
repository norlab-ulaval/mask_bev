#version 330 core

layout (location = 0) in vec4 vertLidarHit;
layout (location = 1) in vec3 vertLabelColor;

out vec3 fragPos;
out float fragIntensity;
out float fragRenderMode;
out vec3 fragLabelColor;

uniform mat4 vertProjMat;
uniform mat4 vertViewMat;
uniform mat4 vertModelMat;
uniform float vertPointSize;
uniform float vertRenderMode;

void main()
{
    vec3 pos = vertLidarHit.xyz;

    gl_Position = vertProjMat * vertViewMat * vertModelMat * vec4(pos, 1.0);
    gl_PointSize = vertPointSize;

    vec4 vertPos4 = vertViewMat * vec4(pos, 1.0);

    fragPos = vec3(vertPos4) / vertPos4.w;
    fragIntensity = vertLidarHit.w;
    fragRenderMode = vertRenderMode;
    fragLabelColor = vertLabelColor;
}
