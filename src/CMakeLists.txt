# Find google profiler
find_library(PROFILER_PATH NAMES profiler)
message(WARNING ${PROFILER_PATH})

set(SOURCE_FILES listen.cpp)
add_library(listen_lib ${SOURCE_FILES})

if (USE_PROFILER)
  target_link_libraries(listen_lib PRIVATE OpenMP::OpenMP_C ${PROFILER_PATH})
else()
  target_link_libraries(listen_lib PRIVATE OpenMP::OpenMP_C)
endif()

