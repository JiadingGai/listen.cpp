set(SOURCE_FILES listen.cpp)
# add_executable(listen ${SOURCE_FILES})
add_library(listen_lib ${SOURCE_FILES})
target_link_libraries(listen_lib PRIVATE OpenMP::OpenMP_C)
