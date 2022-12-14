#include "include/engine.h"
// #include <random>

using std::vector;
using engine::Boid, engine::vec2;

const size_t MAX_N_BOIDS  = 100'000;
const size_t INIT_N_BOIDS = 30;

int main() {
    float theta = 0.0;

    engine::Engine eng(MAX_N_BOIDS, INIT_N_BOIDS);
    eng.setRepulsorFollowsCursor(true);
    eng.run([&]() {
        theta += 0.01;
        if (theta > 2*M_PI) theta -= 2*M_PI;
        vec2 attractor_pos = 0.7f * vec2(cos(theta), sin(theta));
        eng.updateAttractor(attractor_pos);
    });
}