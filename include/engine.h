#ifndef ENGINE_H
#define ENGINE_H

#include "engine_impl.h"

#include <functional>

namespace engine {

using engine_impl::Boid;
using glm::vec2;

class Engine {
public:
    Engine(size_t nBoids) : engine_(nBoids) {}
    void run(std::function<void()> mainLoopCallback = [](){}) { engine_.run(mainLoopCallback); }
    void updateAttractor(vec2 newPosition) { engine_.updateAttractor(newPosition); }
    void updateRepulsor( vec2 newPosition) { engine_.updateRepulsor( newPosition); }

private:
    engine_impl::Engine engine_;
};

} // namespace engine

#endif // ENGINE_H