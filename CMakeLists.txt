cmake_minimum_required(VERSION 3.14)
project(Simplelibm)

add_library(simplelibm src/logf.cc)
target_include_directories(simplelibm PUBLIC include)
target_compile_features(simplelibm PRIVATE cxx_std_17)
