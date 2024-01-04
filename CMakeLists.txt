cmake_minimum_required(VERSION 3.28)
project(gaiwhisper)
set (CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_definitions("WHISPER_SOURCE_ROOT=${CMAKE_SOURCE_DIR}")

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
