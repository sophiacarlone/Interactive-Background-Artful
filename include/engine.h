#ifndef ENGINE_H
#define ENGINE_H

#include "engine_impl.h"

#include <functional>

namespace engine {

class Engine {
public:
    void run(std::function<void()> mainLoopCallback) { engine_.run(mainLoopCallback); }
    void update_position(glm::vec2 new_position) { engine_.update_position(new_position); }

private:
    engine_impl::Engine engine_;
};

} // namespace engine

#endif // ENGINE_H