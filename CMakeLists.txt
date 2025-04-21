cmake_minimum_required(VERSION 3.14)
project(lalogf)

add_library(lalogf src/logf.cc)
target_include_directories(lalogf PUBLIC include)
target_compile_features(lalogf PRIVATE cxx_std_17)

option(ENABLE_TESTS "Enable testing" ON)
if (ENABLE_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
