enable_testing()

# Add the following lines to include Google Test
# Set the path to Google Test source directory
set(GTEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../build/_deps/mygoogletest-src)

## Add Google Test include directories
include_directories(${GTEST_DIR}/googletest/include ${GTEST_DIR}/googlemock/include)

# Add an executable for the tests
set(TEST_FILES test.cpp)
add_executable(whisper_tests ${TEST_FILES})

# Link your library with the test executable
target_link_libraries(whisper_tests listen_lib GTest::gtest_main)

# Add the test using add_test()
add_test(NAME my_test COMMAND whisper_tests)

include(GoogleTest)
gtest_discover_tests(whisper_tests)
