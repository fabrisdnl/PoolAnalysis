# Daniele Fabris
cmake_minimum_required(VERSION 3.0.0)
project(PoolAnalysis VERSION 0.1.0 LANGUAGES C CXX)
# Find packages
find_package(OpenCV REQUIRED)
# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)
# Source files
file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*/*.cpp")
# Add executable
add_executable(${PROJECT_NAME} ${SOURCES})
# Link libraries
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
