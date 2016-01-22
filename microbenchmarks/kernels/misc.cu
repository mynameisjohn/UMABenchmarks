#include <cuda_occupancy.h>
#include <cuda_profiler_api.h>

#include <time.h>
#include <stdlib.h>

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
	uint32_t minGridSize( 0 );
	// find best occupancy stuff (not working on windows)
#ifndef _WIN32
	cuOccupancyMaxPotentialBlockSize( &minGridSize.z, &ret.numThreads, (CUfunction) kernel, 0, 0, 0 );
#else
	ret.numThreads = 1024;
#endif

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

void makeData( float * data, uint32_t N, bool normalize )
{
	for ( uint32_t i = 0; i < N; i++ )
		data[i] = ( (float) rand() / (float) RAND_MAX ) * ( normalize ? 1 : N );
}

void incData( float * data, uint32_t N )
{
	for ( uint32_t i = 0; i < N; i++ )
	{
		data[i] ++;
	}
}

void incSubset( float * data, uint32_t N, int3 S )
{
	for ( uint32_t i = 0; i < N; i++ )
	{
		if ( contains( S, i ) )
		{
			data[i] ++;
		}
	}
}

float touchData( float * data, uint32_t N )
{
	float x = 0;
	for ( uint32_t i = 0; i < N; i++ ) 
	{
		x = data[i];
		data[i] = x;
	}

	return x;
}

float touchSubset( float * data, uint32_t N, int3 S )
{
	float x = 0;
	for ( uint32_t i = 0; i < N; i++ )
	{
		if ( contains( S, i ) )
		{
			x = data[i];
			data[i] = x;
		}
	}

	return x;
}

int3 getRandomSubset(uint32_t N)
{
	int3 subset;
	subset.x = (int) ( ( (float) rand() / (float) RAND_MAX ) * N );
	subset.y = (int) ( ( (float) rand() / (float) RAND_MAX ) * N );
	subset.z = (int) ( ( (float) rand() / (float) RAND_MAX ) * N );
	return subset;
}