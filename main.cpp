#include "include/engine.h"
// #include <random>

using std::vector;
using engine::Boid, engine::vec2;

const size_t N_BOIDS = 30;

int main() {
    // vec2 attractor_pos(0.0);

    engine::Engine eng(N_BOIDS);
    eng.run();
}