#include "util.h"
#include "microbenchmarks.h"

__global__
void subset_G( float * data, int N, int3 subset )
{
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	if ( idx < N && contains( subset, idx ) ) data[idx]++;
}

float SGACFunc::runUMA( uint32_t N, uint32_t dim, uint32_t nIt )
{
	// Create timing objects, do not start
	float timeTaken( 0 );
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop );

	// Determine random subset
	int3 subset = getRandomSubset( N );

	// Allocate data
	size_t size = sizeof( float )*N;
	float *d_Data( 0 );
	cudaMallocManaged( (void **) &d_Data, size );

	// Get max occupancy values
	LaunchParams occ = GetBestOccupancy( (void *) subset_G, N );

	// Start timing
	cudaEventRecord( start );

	// Make random input between 0 and N
	makeData( d_Data, N );

	// Copy to device and back, then touch everything on host
	for ( int i = 0; i<nIt; i++ )
	{
		subset_G << < occ.numBlocks, occ.numThreads >> >( d_Data, N, subset );
		cudaDeviceSynchronize();
		touchData( d_Data, N );

		// reset subset
		subset = getRandomSubset( N );
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

float SGACFunc::runHD( uint32_t N, uint32_t dim, uint32_t nIt )
{
	// Create timing objects, do not start
	float timeTaken( 0 );
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop );

	// Determine random subset
	int3 subset = getRandomSubset( N );

	// Allocate data
	size_t size = sizeof( float )*N;
	float * h_Data( 0 ), *d_Data( 0 );
	h_Data = (float *) malloc( size );
	cudaMalloc( (void **) &d_Data, size );

	// Get max occupancy values
	LaunchParams occ = GetBestOccupancy( (void *) subset_G, N );

	// Start timing
	cudaEventRecord( start );

	// Create random data between 0 and N
	makeData( h_Data, N );

	// Copy to device and back, then touch everything on host
	for ( int i = 0; i<nIt; i++ )
	{
		cudaMemcpy( d_Data, h_Data, size, cudaMemcpyHostToDevice );
		subset_G << <occ.numBlocks, occ.numThreads >> >( d_Data, N, subset );
		cudaMemcpy( h_Data, d_Data, size, cudaMemcpyDeviceToHost );
		touchData( h_Data, N );

		// reset subset
		subset = getRandomSubset( N );
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