cmake_minimum_required(VERSION 3.28)
project(gaiwhisper)
set (CMAKE_CXX_STANDARD 17)

add_executable(listen listen.cpp)
add_test(NAME sanity_check COMMAND listen)
