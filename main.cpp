#include <glm/glm.hpp>
#include "include/engine.h"

using glm::vec2;
using std::vector;

int main() {
    // @todo ??? doesn't draw every second boid
    vector<vec2> boid_positions = {
        vec2(0.0f,  0.0f),
        vec2(0.3f,  0.0f),
        vec2(0.5f, -0.3f),
    };

    engine::Engine eng;
    eng.run([&](){
        eng.update_boids(boid_positions);
    });
}