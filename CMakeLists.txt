cmake_minimum_required(VERSION 3.14)
project(Simplelibm)

add_library(simplelibm src/logf.cc)
target_include_directories(simplelibm PUBLIC include)
target_compile_features(simplelibm PRIVATE cxx_std_17)

option(ENABLE_TESTS "Enable testing" ON)
if (ENABLE_TESTS)
  enable_testing()
  find_package(GTest REQUIRED)
  include(GoogleTest)
  add_subdirectory(tests)
endif()
