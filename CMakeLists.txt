cmake_minimum_required(VERSION 3.28)
project(gaiwhisper)
add_executable(listen listen.cpp)
add_test(NAME sanity_check COMMAND listen)
