#include "util.h"
#include "microbenchmarks.h"

__global__
void subset_G_Rand( float * in, float * out, int N, float thresh )
{
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	if ( idx < N )
	{
		float val = in[idx];
		if ( val < thresh ) out[idx] = val;
	}
}

float SGACRFunc::runUMA( uint32_t N, uint32_t dim, uint32_t nIt )
{
	// Create timing objects, do not start
	float timeTaken( 0 );
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop );

	// data size
	size_t size = sizeof( float ) * N;

	// separate numbers using threshold
	float thresh = (float) rand() / (float) RAND_MAX;

	// Get max occupancy values
	LaunchParams occ = GetBestOccupancy( (void *)subset_G_Rand, N );

	// Allocate and create data
	float *d_In( 0 ), *d_Out( 0 );
	cudaMallocManaged( (void **) &d_In, size );
	cudaMallocManaged( (void **) &d_Out, size );

	// Start timing
	cudaEventRecord( start );

	// Make random input
	for ( int j = 0; j < N; j++ )
	{
		// All should be between 0 and 1
		d_In[j] = (float) rand() / (float) RAND_MAX;
		d_Out[j] = (float) rand() / (float) RAND_MAX;
	}

	// Iterate
	for ( int i = 0; i < nIt; i++ )
	{
		subset_G_Rand << <occ.numBlocks, occ.numThreads >> >( d_In, d_Out, N, thresh );
		cudaDeviceSynchronize();
		for ( int j = 0; j < N; j++ )
		{
			d_In[j]++;
			d_In[j]++;
		}
	}

	// Stop Timing
	cudaEventRecord( stop );
	cudaEventSynchronize( stop );

	// Get elapsed time
	cudaEventElapsedTime( &timeTaken, start, stop );

	// Free
	cudaFree( d_In );
	cudaFree( d_Out );

	return timeTaken;
}

float SGACRFunc::runHD( uint32_t N, uint32_t dim, uint32_t nIt )
{
	// Create timing objects, do not start
	float timeTaken( 0 );
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop );

	// data size
	size_t size = sizeof( float ) * N;

	// separate numbers using threshold
	float thresh = (float) rand() / (float) RAND_MAX;

	// Get max occupancy values
	LaunchParams occ = GetBestOccupancy( (void *)subset_G_Rand, N );

	// Allocate and create data
	float * h_In( 0 ), *h_Out( 0 ), *d_In( 0 ), *d_Out( 0 );
	h_In = (float *) malloc( size );
	h_Out = (float *) malloc( size );
	cudaMalloc( (void **) &d_In, size );
	cudaMalloc( (void **) &d_Out, size );

	// Start timing
	cudaEventRecord( start );

	// Make random input
	for ( int j = 0; j < N; j++ )
	{
		// All should be between 0 and 1
		h_In[j] = (float) rand() / (float) RAND_MAX;
		h_Out[j] = (float) rand() / (float) RAND_MAX;
	}

	// Iterate
	for ( int i = 0; i < nIt; i++ )
	{
		cudaMemcpy( d_In, h_In, size, cudaMemcpyHostToDevice );
		cudaMemcpy( d_Out, h_Out, size, cudaMemcpyHostToDevice );
		subset_G_Rand << <occ.numBlocks, occ.numThreads >> >( d_In, d_Out, N, thresh );
		cudaMemcpy( h_In, d_In, size, cudaMemcpyDeviceToHost );
		cudaMemcpy( h_Out, d_Out, size, cudaMemcpyDeviceToHost );
		for ( int j = 0; j < N; j++ )
		{
			h_In[j]++;
			h_Out[j]++;
		}
	}

	// Stop Timing
	cudaEventRecord( stop );
	cudaEventSynchronize( stop );

	// Get elapsed time
	cudaEventElapsedTime( &timeTaken, start, stop );

	// Free
	free( h_In );
	free( h_Out );
	cudaFree( d_In );
	cudaFree( d_Out );

	return timeTaken;
}
