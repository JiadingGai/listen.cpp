cmake_minimum_required(VERSION 3.18)
project(cuda_example LANGUAGES CUDA CXX)

# Enable CUDA
find_package(CUDA REQUIRED)

# Add CUDA example executable
add_executable(binary_instrumentation binary_instrumentation.cu)

# Set CUDA architecture flags (optional, but recommended)
set_target_properties(binary_instrumentation PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD 17
)

# Link necessary CUDA libraries
target_link_libraries(binary_instrumentation PRIVATE CUDA::cudart)
