add_executable(logf src/main.c)
target_link_libraries(logf lalogf)

add_executable(logf_avx512_test src/logf_avx512_test.cc)
target_link_libraries(logf_avx512_test lalogf gtest::gtest)
gtest_discover_tests(logf_avx512_test)

add_test(NAME ulp_test
         COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/test.py)
