cmake_minimum_required(VERSION 2.8)
project( test )
add_executable( test main.cpp )

find_package( OpenCV REQUIRED )
find_package( glfw3 REQUIRED )
find_package( Vulkan REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( test
    ${OpenCV_LIBS}
    glfw
    vulkan
)


add_custom_command(
    OUTPUT shaders/shader.vert shaders/shader.frag
    COMMAND glslc shaders/shader.vert -o shaders/vert.spv
    COMMAND glslc shaders/shader.frag -o shaders/frag.spv
)
