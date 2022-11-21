#ifndef ENGINE_H
#define ENGINE_H

#include "engine_impl.h"

#include <functional>

namespace engine {

using engine_impl::Boid;
using glm::vec2;

class Engine {
public:
    void run(std::function<void()> mainLoopCallback) { engine_.run(mainLoopCallback); }
    void update_position(vec2 new_position) { engine_.update_attractor(new_position); }
    void update_boids(const std::vector<Boid>& boids) { engine_.update_boids(boids); }

private:
    engine_impl::Engine engine_;
};

} // namespace engine

#endif // ENGINE_H