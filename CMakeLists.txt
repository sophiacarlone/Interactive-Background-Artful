cmake_minimum_required(VERSION 3.13)
project( Interactive-Background-Artful )

# test tracker and boids engine together
add_executable( test test.cpp include/tracker.cpp )
find_package( OpenCV REQUIRED )
find_package( glfw3 REQUIRED )
find_package( Vulkan REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( test
    ${OpenCV_LIBS}
    glfw
    vulkan
)

# test just boids engine
add_executable( test-boids test-boids.cpp )
find_package( glfw3 REQUIRED )
find_package( Vulkan REQUIRED )
target_link_libraries( test-boids
    glfw
    vulkan
)

# test just tracker
add_executable( test-tracker test-tracker.cpp include/tracker.cpp )
find_package( OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( test-tracker
    ${OpenCV_LIBS}
)

# shaders
add_custom_command(
    OUTPUT vert.spv frag.spv comp.spv
    COMMAND glslc ${CMAKE_SOURCE_DIR}/shaders/shader.vert -o vert.spv
    COMMAND glslc ${CMAKE_SOURCE_DIR}/shaders/shader.frag -o frag.spv
    COMMAND glslc ${CMAKE_SOURCE_DIR}/shaders/shader.comp -o comp.spv
    DEPENDS shaders/shader.vert shaders/shader.frag shaders/shader.comp
)
add_custom_target( shaders
    DEPENDS vert.spv frag.spv comp.spv
)
add_dependencies( test shaders )
add_dependencies( test-boids shaders )