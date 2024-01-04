cmake_minimum_required(VERSION 3.28)
project(gaiwhisper)

# Find OpenMP package
# FIRST NEED TO: export OpenMP_ROOT=$(brew --prefix)/opt/libomp
# https://gist.github.com/scivision/16c2ca1dc250f54d34f1a1a35596f4a0
find_package(OpenMP REQUIRED)

set (CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_definitions("WHISPER_SOURCE_ROOT=${CMAKE_SOURCE_DIR}")

# Add -Wall flag to compiler options
add_compile_options(-Wall)

include(FetchContent)

# Fetch GoogleTest using Git
FetchContent_Declare(
  mygoogletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG main # Replace 'main' with the desired branch or commit hash
)

FetchContent_MakeAvailable(mygoogletest)

# Include directories
include_directories(src)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(test)
