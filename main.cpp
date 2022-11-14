#include "include/engine.h"

using std::vector;
using engine::Boid;

int main() {
    // even ones on the left, odd ones on the right
    vector<Boid> boids = {
        Boid { {-0.5f, -0.5f} },
        Boid { { 0.5f, -0.5f} },
        Boid { {-0.5f, -0.4f} },
        Boid { { 0.5f, -0.4f} },
        Boid { {-0.5f, -0.3f} },
        Boid { { 0.5f, -0.3f} },
        Boid { {-0.5f, -0.2f} },
        Boid { { 0.5f, -0.2f} },
        Boid { {-0.5f, -0.1f} },
        Boid { { 0.5f, -0.1f} },
    };

    engine::Engine eng;
    eng.run([&](){
        eng.update_boids(boids);
    });
}