cmake_minimum_required(VERSION 2.6)

project (MICROBENCHMARK)

set (UMA 0 CACHE BOOL "UMA on/off")

if(${UMA})
	add_definitions(-DUMA)
	set(BIN_DIR bin_UMA CACHE PATH "Output dir")
else()
	set(BIN_DIR bin CACHE PATH "Output dir")
endif(${UMA})

find_package(CUDA QUIET REQUIRED)

include_directories(kernels/)

cuda_add_executable(
	${BIN_DIR}/gsRelax
	kernels/gsRelax.cu
)
cuda_add_executable(
	${BIN_DIR}/AllGpuAllCpu
	kernels/AllGpuAllCpu.cu
)
cuda_add_executable(
	${BIN_DIR}/AllGpuSubCpu
	kernels/AllGpuSubCpu.cu
)
cuda_add_executable(
	${BIN_DIR}/SubGpuAllCpu
	kernels/SubGpuAllCpu.cu
)
cuda_add_executable(
	${BIN_DIR}/SubGpuAllCpu_r
	kernels/SubGpuAllCpu_r.cu
)
