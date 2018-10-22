#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;

layout(location = 0) out vec4 varying_color;

layout(binding = 0) uniform UniformBlock {
    mat4 model;
    mat4 view;
    mat4 projection;
} ubo;

layout(push_constant) uniform PushConstants {
    vec4 tint;
    vec3 position;
} push_constants;


void main() {
    varying_color = color * push_constants.tint;
    gl_Position = ubo.projection * ubo.view * ubo.model * vec4(position + push_constants.position, 1.0);
}
