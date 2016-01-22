#pragma once

#include <stdlib.h>
#include <stdint.h>
#include <string>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

class CudaStopWatch
{
	std::string name;
	cudaEvent_t start, stop;
public:
	CudaStopWatch( std::string n ) :
		name( n )
	{
		cudaEventCreate( &start );
		cudaEventCreate( &stop );
		cudaEventRecord( start );
	}

	~CudaStopWatch()
	{
		cudaEventRecord( stop );
		cudaEventSynchronize( stop );

		// Print out the elapsed time
		float mS( 0.f );
		cudaEventElapsedTime( &mS, start, stop );
		printf( "%s took %f mS to execute\n", name.c_str(), mS );
	}
};

template <typename T>
T min( T a, T b )
{
	return ( a < b ? a : b );
}
template <typename T>
T max( T a, T b )
{
	return ( a > b ? a : b );
}
template <typename T>
T min( T a, T m, T M )
{
	return min( max( a, m ), M );
}


template <typename T>
inline void swap( T& a, T& b )
{
	T c( a );
	a = b;
	b = c;
}

__inline__ __host__ __device__
bool contains( int3 s, int e )
{
	return ( s.x == e || s.y == e || s.z == e );
}

__global__
void inc( float * data, int N );

void makeData( float * data, uint32_t N, bool normalize = false );
void incData( float * data, uint32_t N );
void incSubset( float * data, uint32_t N, int3 S);
float touchData( float * data, uint32_t N );
float touchSubset( float * data, uint32_t N, int3 S );
int3 getRandomSubset( uint32_t N );