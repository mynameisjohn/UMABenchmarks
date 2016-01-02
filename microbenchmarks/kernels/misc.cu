#include <cuda_occupancy.h>

#include "microbenchmarks.h"

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