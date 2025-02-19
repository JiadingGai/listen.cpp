cmake_minimum_required(VERSION 3.18)
project(cuda_examples LANGUAGES CUDA CXX)

# Enable CUDA
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

if (CUDAToolkit_FOUND)
  message(STATUS "CUDA found on this system.")
endif()
message(STATUS "${CUDAToolkit_FOUND}")
## DEBUG: print all the variables that starts with "CUDAToolkit_", found by the above find_package(CUDAToolkit REQUIRED)
if (ON)
  get_cmake_property(_variableNames VARIABLES)
  foreach (_variableName ${_variableNames})
    if(_variableName MATCHES "^CUDAToolkit_")
      message(STATUS "${_variableName}=${${_variableName}}")
    endif()
  endforeach()
endif()

# Set GoogleTest and CUTLASS include directories from the submodules
set(GOOGLETEST_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/googletest/googletest/include)
set(CUTLASS_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/cutlass/include)

# Detect GPU architecture dynamically
execute_process(
    COMMAND ${CUDAToolkit_TARGET_DIR}/bin/__nvcc_device_query
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
# target_link_libraries(binary_instrumentation PRIVATE CUDA::cudart)

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
  ${CMAKE_BINARY_DIR}/lib/libgtest.a
)

# Add sgemm_sm80_cris example
add_executable(sgemm_sm80_cris sgemm_sm80_cris.cu)

target_include_directories(sgemm_sm80_cris PRIVATE
    ${CUTLASS_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/cutlass/tools/util/include
    ${CMAKE_SOURCE_DIR}/cutlass/test/unit/gemm/device
    ${GOOGLETEST_INCLUDE_DIR}
)

# Set CUDA architecture flags dynamically
foreach(target binary_instrumentation halfA_int8_mixed sgemm_sm80_cris)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -arch=${COMPUTE_CAPABILITY} -code=${SM_CAPABILITY} -lineinfo --verbose --keep>
    )
endforeach()

