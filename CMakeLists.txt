cmake_minimum_required(VERSION 3.7)

project(CUDATracer LANGUAGES CUDA CXX)

include(CheckLanguage)
check_language(CUDA)

# find_package(SDL2 REQUIRED)
# find_package(OpenGL REQUIRED)
# include_directories(${SDL2_INCLUDE_DIRS})
# include_directories(${OPENGL_INCLUDE_DIR})
add_executable(CUDATracer main.cu)
# target_link_libraries(CUDATracer ${SDL2_LIBRARIES})
