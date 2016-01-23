#pragma once

#include <stdint.h>

// My attempt at mimicking some of the 
// boost::cpu_timer class' functionality
// Under the hood this uses std::chrono::system_clock
// for timing, although it could be templated to use
// different types of clocks
#include <chrono>
struct CpuTimer
{
	using Time = std::chrono::high_resolution_clock;
	using tp = decltype( Time::now() );

private:
	tp m_Start;
	tp m_Stop;
public:
	tp Start();			// Stop timing
	tp Stop();			// Stop timing
	float Elapsed() const;	// Elapsed seconds
};