#version 450

// `location` refers to a framebuffer
// using same framebufs for both; i.e. we're modifying in-place
// @todo does it matter that one is a vec3 and the other is a vec4?
layout(location = 0) in vec3 fragColor; // name doesn't need to be same but it's nice
layout(location = 0) out vec4 outColor;

void main() {
    // we don't need to index into fragColor; the index is implicit
    outColor = vec4(fragColor, 1.0);
}