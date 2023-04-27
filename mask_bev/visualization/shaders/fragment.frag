#version 330 core
precision mediump float;

in vec3 fragPos;
in float fragIntensity;
in float fragRenderMode;
in vec3 fragLabelColor;

out vec4 FragColor;

void main()
{
    vec2 circCoord = 2.0 * gl_PointCoord - 1.0;
    if (dot(circCoord, circCoord) > 2.0) {
        discard;
    }

    if (fragRenderMode < 0.05) {
        FragColor = vec4(fragIntensity * vec3(1, 1, 1), 1);
    } else if (fragRenderMode < 0.15) {
        FragColor = vec4(fragLabelColor, 1);
    } else if (fragRenderMode < 0.25) {
        FragColor = vec4(vec3(0, 0, 0), 1);
    }
}
