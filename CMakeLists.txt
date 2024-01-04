cmake_minimum_required(VERSION 3.28)
project(gaiwhisper)
set (CMAKE_CXX_STANDARD 17)

# Include directories
include_directories(src)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(test)
