from argparse import ArgumentParser
import subprocess as sp
from platform import system
import matplotlib.pyplot as plt
import numpy as np

from Profiler import Profiler
from Benchmarker import Benchmarker

Microbenchmarks = {
	'Relax',	# Gauss Seidel Relaxation
	'SGAC',		# Subset GPU All CPU
	'AGSC',		# All GPU Subset CPU
	'SGACR',	# Subset GPU All CPU Random
	'AGAC'		# All GPU All Cpu
	'All'		# Keyword to run all microbenchmarks
}
#[Profiler(n, ProbSize, NumIt, TestCount) for n in Microbenchmarks]

# set up argparse
parser = ArgumentParser(description = "Use this script to automate the microbenchmarks")

# These should be parsed
ExeName = 'gsRelax'
ProgName = 'Relax'
RunType = 'benchmark'
ProbSize = 2048
Dim = 1
NumIt = 100
Pattern = 'HD'
MakePlots = True
TestCount = 100
Verbose = True

Profilers = []
Benchmarkers = []

# Iterate through all of these, they will be combined into plots
liBenchmarkers.append(Benchmarker('Relax', ProbSize, NumIt, TestCount))
liProfilers.append(Profiler('AGAC',ProbSize, NumIt, Pattern))

# Benchmarkers need to be split up by problem size

# the command formatting can move in class
for bm in liBenchmarkers:
	# format the benchmark command
	bmCmd = [ExeName]
	bmCmd += [ProgName]
	bmCmd += [RunType]
	bmCmd += [str(ProbSize)]
	bmCmd += [str(Dim)]
	bmCmd += [str(NumIt)]
	bmCmd += [str(TestCount)]
	
	# Run the benchmarks
	ret = sp.call(TestCmd, shell = (system() == 'Windows'))
	if (ret != 0):
		continue
	
	# Open Output file
	FileName = ProgName + '_' + str(ProbSize) + '.txt'
	F = open(FileName, 'r')

	# File has two numbers: HD time, UMA time
	runTimes = [float(n) for n in F.readline().split('\t')]
	if (Verbose):
		print('{} took {} mS HD, {} mS UMA'.format(ProgName, runTimes[0], runTimes[1]))

profPlots = []
for prof in liProfilers:
	# nvprof will make this output file
	OutFileName = ProgName + '_' + str(ProbSize) + '.prof'
	# format nvprof command
	nvprofCmd = ['nvprof']
	nvprofCmd += ['--profile-api-trace', 'none']
	nvprofCmd += ['--log-file', OutFileName]
	nvprofCmd += ['--profile-from-start-off']
	nvprofCmd += [ExeName]
	
	# run nvprof, get data, add to list
	ret = sp.call(nvprofCmd, shell = (system() == 'Windows'))
	nvprofData[ProgName].append(ProfileData(OutFileName))
	
	# normalize all profile data, reduce list of data to single datum
	runCount = 1
	nvprofData[ProgName] = sum(nvprofData[ProgName]) / runCount
	
	# Get the bar graph data from the profiler, add to list
	plotData = prof.GetPlotData()
	profPlots.append(plotData)
	
# # Make the command
# TestCmd = [ExeName]
# TestCmd += [ProgName]
# TestCmd += [RunType]
# TestCmd += [str(ProbSize)]
# TestCmd += [str(Dim)]
# TestCmd += [str(NumIt)]
# if (RunType == 'benchmark'):
	# TestCmd += [str(TestCount)]
# elif (RunType == 'profile'):
	# TestCmd += [Pattern]
	
# # Run the benchmarks (shell = True on Windows)
# ret = sp.call(TestCmd, shell = (system() == 'Windows'))
# if (ret != 0):
	# print('Error occured during running test!')
	# print(TestCmd)
	# #return
	
# # Handle benchmark output
# if (RunType == 'benchmark'):
	# # Open Output file
	# FileName = ProgName + '_' + str(ProbSize) + '.txt'
	# F = open(FileName, 'r')

	# # File has two numbers: HD time, UMA time
	# runTimes = [float(n) for n in F.readline().split('\t')]
	# if (Verbose):
		# print('{} took {} mS HD, {} mS UMA'.format(ProgName, runTimes[0], runTimes[1]))
elif (RunType == 'profile'):
	# nvprof will make this output file
	OutFileName = ProgName + '_' + str(ProbSize) + '.prof'
	# format nvprof command
	nvprofCmd = ['nvprof']
	nvprofCmd += ['--profile-api-trace', 'none']
	nvprofCmd += ['--log-file', OutFileName]
	nvprofCmd += ['--profile-from-start-off']
	nvprofCmd += [ExeName]
	
	# run nvprof, get data, add to list
	ret = sp.call(nvprofCmd, shell = (system() == 'Windows'))
	nvprofData[ProgName].append(ProfileData(OutFileName))
	
	# normalize all profile data, reduce list of data to single datum
	runCount = 1
	nvprofData[ProgName] = sum(nvprofData[ProgName]) / runCount
	
	# create plot data
	