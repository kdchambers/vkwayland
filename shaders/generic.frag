#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform sampler2DArray samplerArray;

layout(location = 0) in vec2 inTexCoord;
layout(location = 1) in vec4 inColor;

layout(location = 0) out vec4 outColor;

layout( push_constant ) uniform Block {
    vec2 dimensions;
    float frame;
} PushConstant;

void main() {
    outColor = texture(samplerArray, vec3(inTexCoord, 0)) * inColor;
}
