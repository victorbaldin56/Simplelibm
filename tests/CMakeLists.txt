add_executable(logf src/main.c)
target_link_libraries(logf lalogf)

add_executable(logf_test src/logf_test.cc)
target_link_libraries(logf_test lalogf gtest::gtest)
gtest_discover_tests(logf_test)
