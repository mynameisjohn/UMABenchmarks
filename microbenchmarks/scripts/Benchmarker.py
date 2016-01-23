import subprocess as sp
from functools import reduce
from numbers import Real
from platform import system

class BenchmarkData:
    def __init__(self, *args, **kwargs):
        ctorType = None
        if len(args) is 1:
            if type(args[0]) is str:
                ctorType = 'file'
                OutFileName = args[0]
        elif len(args) is 2:
            if all(isinstance(a, Real) for a in args):
                ctorType = 'val'
        else:
            print('incorrect invocation of BenchmarkData ctor')
            quit()
        if ctorType is 'file':
            F = open(OutFileName, 'r')
            # File has two numbers: HD time, UMA time
            runTimes = [float(n) for n in F.readline().split('\t')]
        
            # Get time
            self.HDTime = runTimes[0]
            self.UMATime = runTimes[1]
        # value constructor
        elif ctorType is 'val':
            self.HDTime = float(args[0])
            self.UMATime = float(args[1])

    def __add__(A, B):
        h = A.HDTime + B.HDTime
        u = A.UMATime + B.UMATime
        return BenchmarkData(h, u)

    def __radd__(self, other):
        return self + other

    def __truediv__(A, s):
        ret = A
        s = float(s)
        ret.HDTime /= s
        ret.UMATime /= s
        return ret
        
class Benchmarker:
    def __init__(self, progName, pSize, nIt, tc):
        self.ProgName = progName
        self.ProbSize = pSize
        self. NumIt = nIt
        self.TestCount = tc
        
        self.__accum = BenchmarkData(0, 0)
        self.__counter = 0
    # This can be called multiple times, no state is remembered
    def Execute(self, ExeName):
        if ExeName is not 'None':
            # format the benchmark command
            bmCmd = [ExeName]
            bmCmd += [self.ProgName]
            bmCmd += ['benchmark']
            bmCmd += [self.ProbSize]
            bmCmd += [1]
            bmCmd += [self.NumIt]
            bmCmd += [self.TestCount]
            bmCmd = [str(x) for x in bmCmd]

            # Run the benchmarks
            ret = sp.call(bmCmd, shell = (system() == 'Windows'))
            if (ret != 0):
                print('Error benchmarking call: ' + str(bmCmd))
                return None
            else:
                print('Successfully benchmarked program {}, N = {}'.format(self.ProgName, self.ProbSize))
        
        # Get output file, return data
        outFile = self.ProgName + '_' + str(self.ProbSize) + '.txt'
        bd = BenchmarkData(outFile)
        
        # maintain accumulator here
        self.__counter += 1
        self.__accum += bd
        return bd
    
    def GetAveragedData(self):
        return self.__accum / float(self.__counter)