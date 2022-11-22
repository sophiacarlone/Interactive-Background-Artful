#include "include/engine.h"
#include <random>

using std::vector;
using engine::Boid, engine::vec2;

const size_t N_BOIDS = 30;

vector<Boid> init_boids(size_t n_boids) {
    // set up RNG stuff
    std::random_device true_rng;
    std::mt19937 generator(true_rng());
    std::uniform_real_distribution<float> rng(-1.0, 1.0);

    vector<Boid> boids(n_boids);
    for (Boid& boid : boids) {
        boid.pos = vec2(rng(generator), rng(generator));
        boid.vel = vec2(0.5); // placeholder value
    }
    return boids;
}

int main() {
    vec2 attractor_pos(0.0);

    engine::Engine eng;
    // @todo engine needs a way to initialize boids before calling run(). The following is a placeholder hack.
    bool first_iter = true;
    eng.run([&](){
        if (first_iter) {
            eng.update_boids(init_boids(N_BOIDS));
            first_iter = false;
        }
    });
}