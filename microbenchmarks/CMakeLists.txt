cmake_minimum_required(VERSION 2.6)

project (MICROBENCHMARK)


find_package(CUDA QUIET REQUIRED)
list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
SET(CMAKE_CXX_FLAGS "-std=c++11")
include_directories(kernels/)

file(GLOB KRN kernels/*.cu)
file(GLOB SRC main.cpp)
file(GLOB INC include/*.h)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

cuda_add_executable(MicroBenchmarks ${SRC} ${KRN} ${INC})
