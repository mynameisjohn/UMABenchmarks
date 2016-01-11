#include "microbenchmarks.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <map>
#include <functional>
#include <memory>

#include <cuda_profiler_api.h>

// Generic error handling
int err( int argc, char ** argv )
{
	printf( "Error! Invalid arguments passed:\n" );
	for ( int i = 0; i < argc; i++ )
		printf( "%s\n", argv[i] );
	return EXIT_FAILURE;
}

int main( int argc, char ** argv )
{
	// See how many args we got
	if ( argc < 7 )
		return err( argc, argv );

	// String based access to those functions
	using PTestFunc = std::unique_ptr < TestFunc >;
	std::map<std::string, PTestFunc> fnMap;
	fnMap.emplace( "SGAC", PTestFunc( new SGACFunc( "SGAC" ) ) );
	fnMap.emplace( "AGSC", PTestFunc( new AGSCFunc( "AGSC" ) ) );
	fnMap.emplace( "SGACR", PTestFunc( new SGACRFunc( "SGACR" ) ) );
	fnMap.emplace( "AGAC", PTestFunc( new AGACFunc( "AGAC" ) ) );
	fnMap.emplace( "Relax", PTestFunc( new RelaxFunc( "Relax" ) ) );

	// See if we have what they want
	std::string progName = argv[1];
	auto it = fnMap.find( progName );
	if ( it == fnMap.end() )
		return err( argc, argv );

	// Get function from map
	TestFunc * fn = it->second.get();

	// Seed random
	srand( time( 0 ) );

	// Get problem size, # iterations, dimension
	int N = atoi( argv[3] );
	int dim = atoi( argv[4] );
	int nIt = atoi( argv[5] );

	if ( N < 0 || dim < 0 || nIt < 0 )
	{
		printf( "Error! Invalid arguments passed:\n" );
		for ( int i = 0; i < argc; i++ )
			printf( "%s\n", argv[i] );
		return EXIT_FAILURE;
	}

	// See if profiling or benchmarking
	std::string type = argv[2];
	if ( type == "profile" )
	{
		ScopedCuProfiler prof;
		std::string pattern = argv[6];
		if ( pattern == "UMA" )
			fn->runUMA( N, dim, nIt );
		else if ( pattern == "HD" )
			fn->runHD( N, dim, nIt );
		else
			return err( argc, argv );

		return EXIT_SUCCESS;
	}
	// We need number of times run for benchmarking
	else if ( type == "benchmark" && argc >= 6 )
	{
		int testCount = atoi( argv[6] );

		// Do both UMA and Host-Device code
		// Create a cuda event, start timing, stop, write to file
		float umaSum( 0 ), hdSum( 0 );
		for ( int i = 0; i < testCount; i++ )
		{
			hdSum += fn->runHD( N, dim, nIt );
			umaSum += fn->runUMA( N, dim, nIt );
		}

		// Find average runtime
		hdSum /= float( testCount );
		umaSum /= float( testCount );

		// Print to file based on prob size
		std::string fileName = fn->GetName();
		fileName.append("_").append( std::to_string(N) ).append( ".txt" );
		FILE * fp = fopen( fileName.c_str(), "w" );
		if ( !fp )
		{
			printf( "Error opening file %s! closing...\n", fileName.c_str() );
			return EXIT_FAILURE;
		}

		fprintf( fp, "%f\t%f", hdSum, umaSum );
		fclose( fp );

		return EXIT_SUCCESS;
	}

	return EXIT_FAILURE;
}
