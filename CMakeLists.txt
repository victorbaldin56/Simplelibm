cmake_minimum_required(VERSION 3.14)
project(lalogf)

add_library(lalogf src/logf.cc)
target_include_directories(lalogf PUBLIC include)
target_compile_features(lalogf PRIVATE cxx_std_17)
target_compile_options(lalogf PUBLIC "-march=native")

option(ENABLE_TESTS "Enable testing" ON)
if (ENABLE_TESTS)
  find_package(GTest REQUIRED)
  enable_testing()
  include(GoogleTest)
  add_subdirectory(tests)
endif()

find_package(benchmark REQUIRED)
add_subdirectory(bench)
