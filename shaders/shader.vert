#version 450

// `location` refers to a framebuffer
// `binding` is used for uniforms

layout(location = 0) in vec2 vertPosition;
layout(location = 1) in vec3 vertColor; // this is currently ignored

struct Boid {
    vec2 pos;
    vec2 vel;
};
// We must ensure the buffer data satisfies std140 alignment requirements, to ensure the strides used by the
// compiled shader aligns with the data.
// See OpenGL 4.6 spec section 7.6.2.2 "Standard Uniform Block Layout" for details.
// Most importantly, the stride over an array will be the base alignment of an element rounded up to the size
// of a vec4.
layout(binding = 0, std140) buffer Boids {
    Boid boids[];
};

// @todo why are we reusing the 0 framebuffer?
layout(location = 0) out vec3 fragColor;

// The `uniform` qualifier is used for data that is passed in from the CPU.
// It's called a "uniform" because it's the same for every invocation in this call (as opposed to, e.g.,
// vertPosition, vertColor, and fragColor, which have a different value for every invocation)
layout(push_constant) uniform PushConsts {
    float BOID_SPEED_MAX;
    float BOID_SPEED_MIN;
};

// radians
vec2 rotate(float ang, vec2 v) {
    /*
        Note the constructor's parameter order:
        the actual matrix is the transpose of the what it looks like in this text format.
        I.e. mat2(
          a, b,    // this line specifies the first COLUMN
          c, d     // this line specifies the second COLUMN
        )
        produces the matrix
          a, c,
          b, d
    */
    mat2 R = mat2(
         cos(ang), sin(ang),
        -sin(ang), cos(ang)
    );

    return R * v;
}

// angle (radians) from the unit X vector (1,0)
float angle(vec2 v) {
    if (v == vec2(0.0f, 0.0f)) return 0; // picked arbitrarily; will claim that the 0-vector orients along the x-axis
    return atan(v.y, v.x); // maybe be suspicious of how this behaves when x or y is 0
}

// expects h in [0,360], s and v each in [0,1]
// reference: https://web.archive.org/web/20221207073025/https://en.wikipedia.org/wiki/HSL_and_HSV
vec3 hsv2rgb(float h, float s, float v) {
    const vec3 n = vec3(5.0, 3.0, 1.0);
    // h *= 360.0; // convert from [0,1] to degrees
    vec3 k = mod((n + h / 60.0), 6.0);
    return v - v * s * max(vec3(0.0), min(min(k, 4.0 - k), vec3(1.0)));
}

void main() {
    // vec3 pos = pushConsts.posTransform * vec3(vertPosition, 1.0f);
    // gl_Position = vec4(pos.xy, 0.0f, 1.0f);
    // gl_Position = pushConsts.posTransform * vec4(vertPosition, 0.0f, 1.0f);

    Boid boid = boids[gl_InstanceIndex];

    // Set the orientation, then the position.
    // The order in which we do the transformations matters.
    float angRad = angle(boid.vel);
    vec2 p = rotate(angRad, vertPosition);
    p += boid.pos;

    gl_Position = vec4(p, 0.0f, 1.0f);

    // linearly map boid velocity from [SPEED_MIN, SPEED_MAX] to [0,1]
    float normalizedVel = (length(boid.vel) - BOID_SPEED_MIN) / (BOID_SPEED_MAX - BOID_SPEED_MIN);
    fragColor = hsv2rgb(degrees(angRad), 0.8, 1.0 - normalizedVel);
}