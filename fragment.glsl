#version 450 core

in vec2 uv;

out vec4 color;

uniform sampler2D tex;

void main()
{
    color = vec4(texture(tex, uv).rgb, 1);
    // color = vec4(1, 0, 0, 1);
}
