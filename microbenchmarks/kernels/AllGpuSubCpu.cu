#include <stdlib.h>

#include "util.h"
#include "microbenchmarks.h"


float AGSCFunc::runUMA( uint32_t N, uint32_t dim, uint32_t nIt )
{
	// Create timing objects, do not start
	float timeTaken( 0 );
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop );

	// Create random subset
	int3 subset;
	subset.x = (int) ( ( (float) rand() / (float) RAND_MAX ) * N );
	subset.y = (int) ( ( (float) rand() / (float) RAND_MAX ) * N );
	subset.z = (int) ( ( (float) rand() / (float) RAND_MAX ) * N );

	// Allocate data
	size_t size = sizeof( float ) * N;
	float *d_Data( 0 );
	cudaMallocManaged( (void **) &d_Data, size );

	// Get max occupancy values
	LaunchParams occ = GetBestOccupancy( (void *)inc, N );

	// Start timing
	cudaEventRecord( start );

	// Run kernel, copy back to host, only touch subset on CPU
	for ( int i = 0; i < nIt; i++ )
	{
		inc << < occ.numBlocks, occ.numThreads >> >( d_Data, N );
		cudaDeviceSynchronize();
		for ( int j = 0; j < N; j++ )
			if ( contains( subset, j ) )
				d_Data[j]++;
	}

	// Stop timing
	cudaEventRecord( stop );
	cudaEventSynchronize( stop );

	// Free
	cudaFree( d_Data );

	// Get elapsed time
	cudaEventElapsedTime( &timeTaken, start, stop );

	return timeTaken;
}

float AGSCFunc::runHD( uint32_t N, uint32_t dim, uint32_t nIt )
{
	// Create timing objects, do not start
	float timeTaken( 0 );
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop );

	// Create random subset
	int3 subset;
	subset.x = (int) ( ( (float) rand() / (float) RAND_MAX ) * N );
	subset.y = (int) ( ( (float) rand() / (float) RAND_MAX ) * N );
	subset.z = (int) ( ( (float) rand() / (float) RAND_MAX ) * N );

	// Allocate data
	size_t size = sizeof( float ) * N;
	float * h_Data( 0 ), *d_Data( 0 );
	h_Data = (float *) malloc( size );
	cudaMalloc( (void **) &d_Data, size );

	// Get max occupancy values
	LaunchParams occ = GetBestOccupancy( (void *) inc, N );

	// Start timing
	cudaEventRecord( start );

	// Run kernel, copy back to host, only touch subset on CPU
	for ( int i = 0; i < nIt; i++ )
	{
		cudaMemcpy( d_Data, h_Data, size, cudaMemcpyHostToDevice );
		inc << < occ.numBlocks, occ.numThreads >> >( d_Data, N );
		cudaMemcpy( h_Data, d_Data, size, cudaMemcpyDeviceToHost );
		for ( int j = 0; j < N; j++ )
			if ( contains( subset, j ) )
				h_Data[j]++;
	}

	// Stop timing
	cudaEventRecord( stop );
	cudaEventSynchronize( stop );

	// Free
	free( h_Data );
	cudaFree( d_Data );

	// Get elapsed time
	cudaEventElapsedTime( &timeTaken, start, stop );

	return timeTaken;
}