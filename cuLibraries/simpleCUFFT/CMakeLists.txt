cmake_minimum_required(VERSION 2.6)

project(simpleCUFFT)

# set source files
set(PROJECT_SRC simpleCUFFT.cu)

find_package(CUDA QUIET REQUIRED)

include_directories(/usr/local/cuda/samples/common/inc/)

# build options
set(GENCODE -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_35,code=compute_35)
set(GENCODE ${GENCODE} -gencode=arch=compute_20,code=sm_20)
set(GENCODE ${GENCODE} -gencode=arch=compute_10,code=sm_10)

# debug options
set(CUDA_NVCC_FLAGS_DEBUG "-G;-g")
set(CUDA_NVCC_FLAGS_RELEASE "-O2")

if (MSVC)
  set(CMAKE_CXX_FLAGS_DEBUG "/DEBUG")
  set(CMAKE_CXX_FLAGS_RELEASE "/O2")
else()
  set(CMAKE_CXX_FLAGS_DEBUG "-g")
  set(CMAKE_CXX_FLAGS_RELEASE "-O2")
endif()

cuda_add_executable(simpleCUFFT ${PROJECT_SRC})
cuda_add_cufft_to_target(simpleCUFFT)
