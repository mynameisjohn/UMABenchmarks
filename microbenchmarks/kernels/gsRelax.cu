#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda_profiler_api.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


#include "util.h"
#include "microbenchmarks.h"

const float minRes( 0.1f );
inline unsigned int sqrtToInt( int i )
{
	return (unsigned int) ( sqrt( i + 0.1 ) );
}

__inline__ __host__ __device__
uint32_t get2Didx( uint32_t x, uint32_t y, uint32_t N )
{
	return x + y * N;
}

__global__
void gsRelax_Laplacian2D_even( float * in, float * out, uint32_t N )
{
	uint32_t idx_X = 2 * ( threadIdx.x + blockDim.x*blockIdx.x );
	uint32_t idx_Y = threadIdx.y + blockDim.y*blockIdx.y;

	if ( idx_X > 0 && idx_X < N - 1 && idx_Y > 0 && idx_Y < N - 1 )
	{
		uint32_t idx = get2Didx( idx_X, idx_Y, N );
		uint32_t idx_x1 = get2Didx( idx_X - 1, idx_Y, N );
		uint32_t idx_x2 = get2Didx( idx_X + 1, idx_Y, N );
		uint32_t idx_y1 = get2Didx( idx_X, idx_Y - 1, N );
		uint32_t idx_y2 = get2Didx( idx_X, idx_Y + 1, N );

		float sum =
			in[idx_x1] +
			in[idx_x2] +
			in[idx_y1] +
			in[idx_y2];

		out[idx] = 0.25f * sum;
	}
}

__global__
void gsRelax_Laplacian2D_odd( float * in, float * out, uint32_t N )
{
	uint32_t idx_X = 2 * ( threadIdx.x + blockDim.x*blockIdx.x ) + 1;
	uint32_t idx_Y = threadIdx.y + blockDim.y*blockIdx.y;

	if ( idx_X > 0 && idx_X < N - 1 && idx_Y > 0 && idx_Y < N - 1 )
	{
		uint32_t idx = get2Didx( idx_X, idx_Y, N );
		uint32_t idx_x1 = get2Didx( idx_X - 1, idx_Y, N );
		uint32_t idx_x2 = get2Didx( idx_X + 1, idx_Y, N );
		uint32_t idx_y1 = get2Didx( idx_X, idx_Y - 1, N );
		uint32_t idx_y2 = get2Didx( idx_X, idx_Y + 1, N );

		float sum =
			in[idx_x1] +
			in[idx_x2] +
			in[idx_y1] +
			in[idx_y2];

		out[idx] = 0.25f * sum;
	}
}

__global__
void gsRelax_Laplacian1D_even( float * in, float * out, uint32_t N)
{
	uint32_t idx = 2 * ( threadIdx.x + blockDim.x*blockIdx.x );

	if ( idx > 0 && idx < N - 1 )
	{
		float sum = in[idx - 1] + in[idx + 1];
		out[idx] = 0.5f * sum;
	}
}

__global__
void gsRelax_Laplacian1D_odd( float * in, float * out, uint32_t N )
{
	uint32_t idx = 2 * ( threadIdx.x + blockDim.x*blockIdx.x ) + 1;

	if ( idx > 0 && idx < N - 1 )
	{
		float sum = in[idx - 1] + in[idx + 1];
		out[idx] = 0.5f * sum;
	}
}


inline float getResidueSq( float * in, float * out, uint32_t N )
{
	float r( 0 );
	for ( uint32_t i = 0; i<N; i++ )
		r += pow( out[i] - in[i], 2 );
	return r;
}


float RelaxFunc::runUMA( uint32_t N, uint32_t dim, uint32_t nIt )
{
	// Create timing objects, do not start
	CpuTimer T;

	//Just a stupid pad
	dim = ( dim % 2 ? 1 : 2 );
	N = ( dim == 1 ? N : N*N );
	uint32_t size = sizeof( float )*N;
	float * d_Data_A( 0 ), *d_Data_B( 0 );
	float res( 0 );
	cudaMallocManaged( (void **) &d_Data_A, size );
	cudaMallocManaged( (void **) &d_Data_B, size );

	// Get max occupancy values
	LaunchParams occ = GetBestOccupancy( dim == 2 ? (void *) gsRelax_Laplacian1D_even : (void *) gsRelax_Laplacian2D_even, N );

	// Start timing
	T.Start();	

	// Create random data
	makeData( d_Data_A, N, true );
	makeData( d_Data_B, N, true);

	//Repetitive, I know, but I didn't want to introduce overhead during iteration
	if ( dim == 1 )
	{
		for ( int i = 0; i < nIt; i++ )
		{
			gsRelax_Laplacian1D_even << <occ.numBlocks, occ.numThreads >> >( d_Data_A, d_Data_B, N );
			gsRelax_Laplacian1D_odd << < occ.numBlocks, occ.numThreads >> >( d_Data_A, d_Data_B, N );
			cudaDeviceSynchronize();
			res = sqrt( getResidueSq( d_Data_A, d_Data_B, N ) );

			// I'm not really sure why I'm swapping these...
			swap( d_Data_A, d_Data_B );
		}
	}
	else if ( dim == 2 )
	{
		// Right now it's assumed there's a nice sqrt of numThreads (usually 1024 ==> 32x32)
		uint32_t len = sqrt( N );
		uint3 numThreads = make_uint3( sqrtToInt( occ.numThreads ), sqrtToInt( occ.numThreads ), 0 );
		uint3 numBlocks = make_uint3( len / occ.numThreads, 1, 0 ); // not sure about this
		for ( int i = 0; i < nIt; i++ )
		{
			gsRelax_Laplacian2D_even << <numBlocks, numThreads >> >( d_Data_A, d_Data_B, len );
			gsRelax_Laplacian2D_odd << <numBlocks, numThreads >> >( d_Data_A, d_Data_B, len );
			cudaDeviceSynchronize();
			res = sqrt( getResidueSq( d_Data_A, d_Data_B, N ) );

			swap( d_Data_A, d_Data_B );
		}
	}

	// Get elapsed time
	cudaThreadSynchronize();
	float timeTaken = T.Elapsed();

	// Free data
	cudaFree( d_Data_A );
	cudaFree( d_Data_B );

	return timeTaken;
}

float RelaxFunc::runHD( uint32_t N, uint32_t dim, uint32_t nIt )
{
	// Create timing objects, do not start
	CpuTimer T;

	//Just a stupid pad
	dim = ( dim % 2 ? 1 : 2 );
	N = ( dim == 1 ? N : N*N );
	uint32_t size = sizeof( float )*N;
	float * h_Data_A( 0 ), *d_Data_A( 0 ), *h_Data_B( 0 ), *d_Data_B( 0 );
	float res( 0 );
	h_Data_A = (float *) malloc( size );
	h_Data_B = (float *) malloc( size );
	cudaMalloc( (void **) &d_Data_A, size );
	cudaMalloc( (void **) &d_Data_B, size );

	// Get max occupancy values
	LaunchParams occ = GetBestOccupancy( dim == 2 ? (void *) gsRelax_Laplacian1D_even : (void *) gsRelax_Laplacian2D_even, N );

	// Start timing
	T.Start();

	// Make random data between 0 and 1
	makeData( h_Data_A, N, true );
	makeData( h_Data_B, N, true );

	//Repetitive, I know, but I didn't want to introduce overhead during iteration
	if ( dim == 1 )
	{
		//CudaStopWatch CSW( "UMA" );
		//int nT( 1024 ), nB( ( N / 1024 ) / 2 + 1 );
		for ( int i = 0; i < nIt; i++ )
		{
			cudaMemcpy( d_Data_A, h_Data_A, size, cudaMemcpyHostToDevice );
			cudaMemcpy( d_Data_B, h_Data_B, size, cudaMemcpyHostToDevice );
			gsRelax_Laplacian1D_even << < occ.numBlocks, occ.numThreads >> >( d_Data_A, d_Data_B, N );
			gsRelax_Laplacian1D_odd << < occ.numBlocks, occ.numThreads >> >( d_Data_A, d_Data_B, N );
			cudaMemcpy( h_Data_A, d_Data_A, size, cudaMemcpyDeviceToHost );
			cudaMemcpy( h_Data_B, d_Data_B, size, cudaMemcpyDeviceToHost );
			res = sqrt( getResidueSq( h_Data_A, h_Data_B, N ) );

			swap( h_Data_A, h_Data_B );
			swap( d_Data_A, d_Data_B );
		}
	}
	else if ( dim == 2 )
	{
		// Right now it's assumed there's a nice sqrt of numThreads (usually 1024 ==> 32x32)
		uint32_t len = sqrt( N );
		uint3 numThreads = make_uint3( sqrtToInt( occ.numThreads ), sqrtToInt( occ.numThreads ), 0 );
		uint3 numBlocks = make_uint3( len / occ.numThreads, 1, 0 ); // not sure about this
		for ( int i = 0; i < nIt; i++ )
		{
			cudaMemcpy( d_Data_A, h_Data_A, size, cudaMemcpyHostToDevice );
			cudaMemcpy( d_Data_B, h_Data_B, size, cudaMemcpyHostToDevice );
			gsRelax_Laplacian2D_even << <numBlocks, numThreads >> >( d_Data_A, d_Data_B, len );
			gsRelax_Laplacian2D_odd << <numBlocks, numThreads >> >( d_Data_A, d_Data_B, len );
			cudaMemcpy( h_Data_A, d_Data_A, size, cudaMemcpyDeviceToHost );
			cudaMemcpy( h_Data_B, d_Data_B, size, cudaMemcpyDeviceToHost );
			res = sqrt( getResidueSq( h_Data_A, h_Data_B, N ) );

			swap( h_Data_A, h_Data_B );
			swap( d_Data_A, d_Data_B );
		}
	}

	// Get elapsed time
	cudaThreadSynchronize();
	float timeTaken = T.Elapsed();

	// Free data
	free( h_Data_A );
	free( h_Data_B );
	cudaFree( d_Data_A );
	cudaFree( d_Data_B );

	return timeTaken;
}