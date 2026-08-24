cmake_minimum_required(VERSION 3.20)
project(edgegenbench_native VERSION 0.1.0 LANGUAGES CXX)

option(EDGEBENCH_BUILD_TESTS "Build native unit tests" ON)
option(EDGEBENCH_BUILD_TOOLS "Build native benchmark CLI" ON)
option(EDGEBENCH_ENABLE_ORT "Enable ONNX Runtime backend" OFF)

add_library(edgegenbench_runtime
  src/config.cpp
  src/model_contract.cpp
  src/preprocess.cpp
  src/session.cpp)
target_include_directories(edgegenbench_runtime PUBLIC include)
target_compile_features(edgegenbench_runtime PUBLIC cxx_std_17)
target_compile_options(edgegenbench_runtime PRIVATE -Wall -Wextra -Wpedantic)

if(EDGEBENCH_ENABLE_ORT)
  if(NOT DEFINED ONNXRUNTIME_ROOT)
    message(FATAL_ERROR "Set ONNXRUNTIME_ROOT when EDGEBENCH_ENABLE_ORT=ON")
  endif()
  target_sources(edgegenbench_runtime PRIVATE src/ort_session.cpp)
  target_include_directories(edgegenbench_runtime PRIVATE "${ONNXRUNTIME_ROOT}/include")
  target_link_directories(edgegenbench_runtime PRIVATE "${ONNXRUNTIME_ROOT}/lib")
  target_link_libraries(edgegenbench_runtime PRIVATE onnxruntime)
  target_compile_definitions(edgegenbench_runtime PUBLIC EDGEBENCH_ENABLE_ORT=1)
endif()

if(EDGEBENCH_BUILD_TOOLS)
  add_executable(edgegenbench_benchmark apps/benchmark.cpp)
  target_link_libraries(edgegenbench_benchmark PRIVATE edgegenbench_runtime)
endif()

if(EDGEBENCH_BUILD_TESTS)
  enable_testing()
  add_executable(edgegenbench_tests tests/runtime_test.cpp)
  target_link_libraries(edgegenbench_tests PRIVATE edgegenbench_runtime)
  add_test(NAME edgegenbench_tests COMMAND edgegenbench_tests)
endif()
