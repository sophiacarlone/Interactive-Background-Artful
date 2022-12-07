#version 450

// `location` refers to a framebuffer
// `binding` is used for uniforms

layout(location = 0) in vec2 vertPosition;
layout(location = 1) in vec3 vertColor; // currently ignoring this

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

void main() {
    // vec3 pos = pushConsts.posTransform * vec3(vertPosition, 1.0f);
    // gl_Position = vec4(pos.xy, 0.0f, 1.0f);
    // gl_Position = pushConsts.posTransform * vec4(vertPosition, 0.0f, 1.0f);

    Boid boid = boids[gl_InstanceIndex];

    // Set the orientation, then the position.
    // The order in which we do the transformations matters.
    vec2 p = rotate(angle(boid.vel), vertPosition);
    p += boid.pos;

    gl_Position = vec4(p, 0.0f, 1.0f);

    // linearly map boid velocity from [SPEED_MIN, SPEED_MAX] to [0,1]
    float normalizedVel = (length(boid.vel) - BOID_SPEED_MIN) / (BOID_SPEED_MAX - BOID_SPEED_MIN);
    // color the boid a shade of gray that scales with its velocity
    fragColor = normalizedVel * vec3(1.0);
}