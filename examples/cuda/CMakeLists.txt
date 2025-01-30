cmake_minimum_required(VERSION 3.18)
project(cuda_examples LANGUAGES CUDA CXX)

# Enable CUDA
find_package(CUDA REQUIRED)

# Set GoogleTest and CUTLASS include directories from the submodules
set(GOOGLETEST_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/googletest/googletest/include)
set(CUTLASS_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/cutlass/include)

# Detect GPU architecture dynamically
execute_process(
    COMMAND ${CUDA_NVCC_EXECUTABLE} -arch=sm_00 --run "cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0); printf(\"%d%d\\n\", prop.major, prop.minor);"
    OUTPUT_VARIABLE GPU_ARCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(GPU_ARCH)
    set(COMPUTE_CAPABILITY "compute_${GPU_ARCH}")
    set(SM_CAPABILITY "sm_${GPU_ARCH}")
else()
    message(WARNING "Could not detect GPU architecture, defaulting to compute_80 and sm_80")
    set(COMPUTE_CAPABILITY "compute_80")
    set(SM_CAPABILITY "sm_80")
endif()

message(STATUS "Detected GPU architecture: ${GPU_ARCH}")

# Add binary_instrumentation example
add_executable(binary_instrumentation binary_instrumentation.cu)
set_target_properties(binary_instrumentation PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD 17
)
target_link_libraries(binary_instrumentation PRIVATE CUDA::cudart)

# Add halfA_int8_mixed example (CUTLASS + GoogleTest)
add_executable(halfA_int8_mixed halfA_int8_mixed.cu)

# Include directories for CUTLASS and GoogleTest
target_include_directories(halfA_int8_mixed PRIVATE
    ${CUTLASS_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/cutlass/tools/util/include
    ${CMAKE_SOURCE_DIR}/cutlass/test/unit/gemm/device
    ${GOOGLETEST_INCLUDE_DIR}
)

# Link GoogleTest from CMAKE_SOURCE_DIR
target_link_libraries(halfA_int8_mixed PRIVATE
    CUDA::cudart
    ${CMAKE_SOURCE_DIR}/googletest/build/lib/libgtest.a
)

# Set CUDA architecture flags dynamically
foreach(target binary_instrumentation halfA_int8_mixed)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -arch=${COMPUTE_CAPABILITY} -code=${SM_CAPABILITY}>
    )
endforeach()

