#version 450

// `location` refers to a framebuffer
// `binding` is used for uniforms

layout(location = 0) in vec2 vertPosition;
layout(location = 1) in vec3 vertColor;
//
// member data must satisfy alignment requirements
layout(binding = 0) uniform Boids {
    // The array size is just the allocation, not necessarily how many we'll render.
    // I picked the number arbitrarily. I think the biggest we can make it is defined by
    // VkPhysicalDeviceLimits::maxUniformBufferRange.
    // We only need vec2s, but the compiled SPIRV ("on my system") uses 16-byte strides, causing every second
    // vec2 to be ignored. I was not able to determine what it's _supposed_ to be from documentation, so I'm
    // just using vec4s here (to guarantee it uses 16 bytes on all systems) and only paying attention to the
    // xy components.
    vec4 pos[1024];
    // @todo add data to represent orientation
} boids;

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

    vec2 p = boids.pos[gl_InstanceIndex].xy + vertPosition;
    // vec2 p = boids.pos[gl_InstanceIndex].xy + vertPosition;

    // the fact that this works suggests that instancing works fine. So something must be wrong with the UBO
    // or its descriptor
    // vec2 p = 0.1 * float(gl_InstanceIndex) + vertPosition;

    gl_Position = vec4(p, 0.0f, 1.0f);
    fragColor = vertColor;
}