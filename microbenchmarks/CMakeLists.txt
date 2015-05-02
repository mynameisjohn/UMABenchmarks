cmake_minimum_required(VERSION 2.6)

project (MICROBENCHMARK)

find_package(CUDA QUIET REQUIRED)

include_directories(kernels/)

cuda_add_executable(
	bin/gsRelax
	kernels/gsRelax.cu
)
cuda_add_executable(
	bin/AllGpuAllCpu
	kernels/AllGpuAllCpu.cu
)
cuda_add_executable(
	bin/AllGpuSubCpu
	kernels/AllGpuSubCpu.cu
)
cuda_add_executable(
	bin/SubGpuAllCpu
	kernels/SubGpuAllCpu.cu
)
cuda_add_executable(
	bin/SubGpuAllCpu_r
	kernels/SubGpuAllCpu_r.cu
)
