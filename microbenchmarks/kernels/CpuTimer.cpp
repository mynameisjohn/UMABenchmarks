#include "CpuTimer.h"

// When the timer starts, the stop time gets set to 0
CpuTimer::tp CpuTimer::Start()
{
	m_Start = Time::now();

	return m_Start;
}

// Assign stop time, return elapsed
CpuTimer::tp CpuTimer::Stop()
{
	m_Stop = Time::now();
	
	return m_Stop;
}

// Return current time - start time
float CpuTimer::Elapsed() const
{
	using std::chrono::duration_cast;
	using TType = std::chrono::nanoseconds;

	// Return in mS
	auto dur = duration_cast<TType>( Time::now() - m_Start );
	return float( dur.count() ) * 1e-6;
}