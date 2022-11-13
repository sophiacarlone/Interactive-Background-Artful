#version 450

// `location` refers to a framebuffer
// `binding` is used for uniforms

layout(location = 0) in vec2 vertPosition;
layout(location = 1) in vec3 vertColor;
//
// member data must satisfy alignment requirements
layout(binding = 0) uniform BoidPositions {
    // The array size is just the allocation, not necessarily how many we'll render.
    // I picked the number arbitrarily. I think the biggest we can make it is defined by
    // VkPhysicalDeviceLimits::maxUniformBufferRange.
    vec2 pos[1024];
    // @todo add data to represent orientation
} boidPositions;

// @todo why are we reusing the 0 framebuffer?
layout(location = 0) out vec3 fragColor;

// The `uniform` qualifier is used for data that is passed in from the CPU.
// It's called a "uniform" because it's the same for every invocation in this call (as opposed to, e.g.,
// vertPosition, vertColor, and fragColor, which have a different value for every invocation)
layout(push_constant) uniform PushConsts {
    mat4 posTransform;
} pushConsts;

void main() {
    // gl_Position is a special variable, so we don't need to specify an output framebuffer
    // vec3 pos = pushConsts.posTransform * vec3(vertPosition, 1.0f);
    // gl_Position = vec4(pos.xy, 0.0f, 1.0f);
    // gl_Position = pushConsts.posTransform * vec4(vertPosition, 0.0f, 1.0f);
    vec2 pos = boidPositions.pos[gl_InstanceIndex] + vertPosition;
    gl_Position = vec4(pos, 0.0f, 1.0f);
    fragColor = vertColor;
}