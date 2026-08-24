cmake_minimum_required(VERSION 3.22)
project(edgegenbench_android LANGUAGES CXX)

set(EDGEBENCH_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(EDGEBENCH_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../../../../native
                 ${CMAKE_CURRENT_BINARY_DIR}/edgegenbench-native)
add_library(edgegenbench_jni SHARED jni_bridge.cpp)
target_link_libraries(edgegenbench_jni PRIVATE edgegenbench_runtime log)
target_compile_features(edgegenbench_jni PRIVATE cxx_std_17)
