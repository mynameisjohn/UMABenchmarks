project(histEqualizationNPP)

set(PROJECT_SRC histEqualizationNPP.cpp)

find_package(CUDA QUIET REQUIRED)

#link_directories(/usr/local/cuda/samples/7_CUDALibraries/common/FreeImage/lib/linux/i686/)

include_directories(/usr/local/cuda/samples/common/inc/ /usr/local/cuda/samples/7_CUDALibraries/common/UtilNPP/ /usr/local/cuda/samples/7_CUDALibraries/common/FreeImage/include)
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

# set runtime library for VC
if (MSVC)
  set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} ";-Xcompiler;/MTd")
  set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG} " /MTd")
  
  set(CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE} ";-Xcompiler;/MT")
  set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE} " /MT")
endif()

cuda_add_executable(histEqualizationNPP ${PROJECT_SRC} OPTIONS ${GENCODE})
target_link_libraries(histEqualizationNPP nppi nppc freeimage)
