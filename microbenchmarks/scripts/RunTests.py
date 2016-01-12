from argparse import ArgumentParser

from Profiler import Profiler
from Benchmarker import Benchmarker
import DataPlot

RunTypes = {
    'bm',
    'p'
}

Microbenchmarks = {
    'Relax',    # Gauss Seidel Relaxation
    'SGAC',        # Subset GPU All CPU
    'AGSC',        # All GPU Subset CPU
    'SGACR',    # Subset GPU All CPU Random
    'AGAC',        # All GPU All Cpu
    'All'        # Keyword to run all microbenchmarks
}
#[Profiler(n, ProbSize, NumIt, TestCount) for n in Microbenchmarks]

# set up argparse
parser = ArgumentParser(description = "Use this script to automate the microbenchmarks")
parser.add_argument('-e', metavar='ExeName', type=str, nargs=1)
parser.add_argument('-p', metavar='Prog', type=str, nargs='+')
parser.add_argument('-t', metavar='Type', type=str, nargs='+')
parser.add_argument('-N', metavar='ProblemSize', type=int, nargs='+')
parser.add_argument('-n', metavar='NumIt', type=int, nargs=1)
parser.add_argument('-a', metavar='AccessPattern', type=str, nargs=1, default='HD')
parser.add_argument('-tc', metavar='TestCount', type=int, nargs=1, default=1)
parser.add_argument('-mp', metavar='MakePlots', type=bool, nargs=1, default=False)
parser.add_argument('-v', metavar='Verbose', type=bool, nargs=1, default=False)
parser.add_argument('-S', metavar='ShowPlots', type=bool, nargs=1, default=False)

# Get args
args = parser.parse_args()
print(args)

# Sanity check
if (set(args.p) & Microbenchmarks == None):
    print('Error: Invalid program name(s) : ' + str(args.p))
    quit()
if (set(args.t) & RunTypes == None):
    print('Error: Invalid run type(s) : ' + str(args.t))
    quit()

# Show matplotlib output as it comes
if args.S:
    DataPlot.g_ShowPlots = True

# If the program name contained all, make a note of it
if ('All' in args.p):
    Microbenchmarks.remove('All')
    args.p = Microbenchmarks

# Create all profiler and benchmarker objects
dProfilers = dict()        # Profilers are indexed by program name (max pSize)
dBenchmarkers = dict()    # Benchmarkers are index [pSize][progName]

# Create all profilers
if ('profile' in args.t):
    for progName in args.p:
        # Each program gets a Profile; just pick biggest prob size
        p = Profiler(progName, max(args.N), args.n[0], args.a)
        dProfilers[progName] = p
        
# Create all benchmarkers
if ('benchmark' in args.t):
    # for every problem size given to us
    for pSize in args.N:
        # if we haven't seen this before
        if pSize not in dBenchmarkers:
            # make a dict
            dBenchmarkers[pSize] = dict()
        # Then loop through every program name given to us
        for progName in args.p:
            # if this program doesn't exist in the psize dict
            if progName not in dBenchmarkers[pSize]:
                # add a new benchmarker to the dict
                b = Benchmarker(progName, pSize, args.n[0], args.tc)
                dBenchmarkers[pSize][progName] = b

# execute all tests
runCount = 1
for i in range(0, runCount):
    # execute all profilers
    for prof in dProfilers.values():
        prof.Execute(args.e[0])
    
    # execute all benchmarkers
    for pSizeDict in dBenchmarkers.values():
        for bm in pSizeDict.values():
            bm.Execute(args.e[0])

# Make plots
if len(dProfilers) > 0:
    DataPlot.MakeProfilerPlot(dProfilers)
    
if len(dBenchmarkers) > 0:
    DataPlot.MakeBenchmarkPlots(dBenchmarkers)