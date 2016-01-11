#include <cuda_occupancy.h>
#include <cuda_profiler_api.h>

#include "microbenchmarks.h"

#include "util.h"

__global__
void inc( float * data, int N )
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if ( idx < N )
		data[idx]++;
}

LaunchParams GetBestOccupancy( void * kernel, int N )
{
	LaunchParams ret = { 0, 0 };
	int minGridSize( 0 );
	// find best occupancy stuff (not working on windows)
	ret.numThreads = 1024;

	// Pick a sensible block number
	ret.numBlocks = ( N + ret.numThreads - 1 ) / ret.numThreads;

	return ret;
}

std::string TestFunc::GetName() const
{
	return m_StrName;
}

ScopedCuProfiler::ScopedCuProfiler()
{
	cudaProfilerStart();
}

ScopedCuProfiler::~ScopedCuProfiler()
{
	cudaProfilerStop();
}
