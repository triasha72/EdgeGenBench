cmake_minimum_required(VERSION 3.22)
project(edgegenbench_android LANGUAGES CXX)

set(EDGEBENCH_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(EDGEBENCH_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
# EDGEBENCH_ENABLE_ORT and ONNXRUNTIME_ROOT are supplied by Gradle only for an
# explicit QNN build. The default CI APK remains dependency-free/reference-only.
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../../../../native
                 ${CMAKE_CURRENT_BINARY_DIR}/edgegenbench-native)
add_library(edgegenbench_jni SHARED jni_bridge.cpp)
target_link_libraries(edgegenbench_jni PRIVATE edgegenbench_runtime log)
target_compile_features(edgegenbench_jni PRIVATE cxx_std_17)
# Android 15 devices may use 16 KiB pages. Keep the explicit linker guard even
# with NDK r27 flexible-page support so CI can verify every PT_LOAD segment.
target_link_options(edgegenbench_jni PRIVATE "-Wl,-z,max-page-size=16384")
