cmake_minimum_required(VERSION 3.13)
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


# also compile the shaders
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