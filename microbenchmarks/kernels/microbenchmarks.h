#pragma once

#include <stdint.h>
#include <string>
#include <cuda_runtime.h>

// More useful for benchmarking than anything else
//int runRelax( int argc, char ** argv );
//int runSubGAllC( int argc, char ** argv );
//int runAllGSubC( int argc, char ** argv );
//int runSubGAllC_Rand( int argc, char ** argv );
//int runAllGAllC( int argc, char ** argv );

struct LaunchParams
{
	uint32_t numBlocks;
	uint32_t numThreads;
};
LaunchParams GetBestOccupancy( void * kernel, int N );

class TestFunc
{
protected:
	std::string m_StrName;
public:
	TestFunc() {}
	TestFunc(std::string name) : m_StrName(name) {}
	virtual float runHD( uint32_t N, uint32_t dim, uint32_t nIt ) = 0;
	virtual float runUMA( uint32_t N, uint32_t dim, uint32_t nIt ) = 0;

	std::string GetName() const;
};

#define __TFInherit(name) \
class name : public TestFunc \
{ \
public:\
	name() : TestFunc(){} \
	name(std::string n) : TestFunc(n){} \
	float runHD( uint32_t N, uint32_t dim, uint32_t nIt ) override; \
	float runUMA( uint32_t N, uint32_t dim, uint32_t nIt ) override; \
};

__TFInherit( RelaxFunc );
__TFInherit( SGACFunc );
__TFInherit( AGSCFunc );
__TFInherit( AGACFunc );
__TFInherit( SGACRFunc );