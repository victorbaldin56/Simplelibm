add_executable(logf_test src/logf_test.cc)
target_link_libraries(logf_test lalogf gtest::gtest boost::boost llvm-openmp::llvm-openmp)
gtest_discover_tests(logf_test)

add_executable(logf_report src/report.cc)
target_link_libraries(logf_report lalogf boost::boost fmt::fmt)
