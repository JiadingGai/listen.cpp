# List your test files here
set(TEST_FILES test.cpp)

# Add an executable for the tests
add_executable(whisper_test ${TEST_FILES})

# Link your library with the test executable
# target_link_libraries(tests_executable your_library)

# Add the test using add_test()
add_test(NAME my_test COMMAND whisper_test)
