cmake_minimum_required(VERSION 3.18)

# Add CUDA example subdirectory
# Find CUDA
find_package(CUDA)

# Check if CUDA was found
if(CUDA_FOUND)
    message(STATUS "CUDA is installed and found on this system.")
    add_subdirectory(cuda)
else()
    message(STATUS "CUDA is not installed or not found on this system.")
endif()
