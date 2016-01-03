from argparse import ArgumentParser
import subprocess as sp
from platform import system

from Profiler import Profiler
from Benchmarker import Benchmarker
from DataPlot import *

RunTypes = {
	'bm',
	'p',
	'both'
}

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
parser.add_argument('ExeName' type=str, nargs=1)
parser.add_argument('Prog', type=str, nargs='+')
parser.add_argument('Type', type=str, nargs=1)
parser.add_argument('PSize', type=int, nargs='+')
parser.add_argument('NumIt', type=int, nargs=1)
parser.add_argument('Pattern', type=str, nargs=1, default='HD')
parser.add_argument('TestCount', type=int, nargs=1, default=1)
parser.add_argument('MakePlots', type=bool, nargs=1, default=False)
parser.add_argument('Verbose', type=bool, nargs=1, default=False)
parser.add_argument('ShowPlots', type=bool, nargs=1, default=False)

# Get args
args = parser.parse_args()

# If the program name contained all, make a note of it
if ('All' in args.Prog):
	args.Prog = Microbenchmarks

# Create all profiler and benchmarker objects
dProfilers = dict()		# Profilers are indexed by program name (max pSize)
dBenchmarkers = dict()	# Benchmarkers are index [pSize][progName]

# Create all profilers
if (args.Type == 'bm' or args.Type == 'both'):
	for progName in args.Prog:
		p = Profiler(progName, max(args.PSize), args.NumIt, args.Pattern)
		dProfilers[progName] = p
		
# Create all benchmarkers
if (args.Type == 'p' or args.Type == 'both'):
	# for every problem size given to us
	for pSize in args.PSize:
		# if we haven't seen this before
		if pSize not in dBenchmarkers:
			# make a dict
			dBenchmarkers[pSize] = dict())
		# Then loop through every program name given to us
		for progName in args.Prog
			# if this program doesn't exist in the psize dict
			if progName not in dBenchmarkers[pSize]:
				# add a new benchmarker to the dict
				b = Benchmarker(progName, args.PSize, args.NumIt, args.TestCount)
				dBenchmarkers[pSize][progName] = b

# execute all tests
runCount = 1
for i in range(0, runCount):
	# execute all profilers
	for prog, prof in dProfilers:
		prof.Execute()
	
	# execute all benchmarkers
	for prog, pSizeDict in dBenchmarkers:
		for psize, bm in pSizeDict:
			bm.Execute()

# Make plots
if len(dProfilers):
	MakeProfilerPlot(dProfilers)
	
if len(dBenchmarkers):
	MakeBenchmarkPlots(dBenchmarkers)