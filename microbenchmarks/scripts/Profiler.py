import subprocess as sp
from platform import system
from numbers import Real
from functools import reduce

class ProfileData:
    def __init__(self, *args, **kwargs):
        ctorType = 'val'
        if len(args) is 1:
            if type(args[0] == str):
                ctorType = 'file'
                OutFileName = args[0]
        elif len(args) is 5:
            if not all(isinstance(a, Real) for a in args):
                print('incorrect invocation')
                ctorType = None
        if ctorType is 'file':
            self.KernelTime = 0.
            self.MemsetTime = 0.
            self.DtoHTime = 0.
            self.HtoDTime= 0.
            self.APITime = 0.
        
            # Parse output file, skip first four lines
            F = open(OutFileName, 'r')
            lines = F.readlines()[4:]
            for line in lines:
                # End of output, I think
                if line[0] == '=' or len(line) == 1:
                    break
            
                # split up data
                data = line.split()
            
                # Get percent and function type
                pct= float(data[0][:-1])
                func = ''.join(data[6:])
            
                # CUDA calls
                if ('CUDA' in func and '[' in func and ']' in func):
                    if 'HtoD' in func:
                        self.HtoDTime += pct
                    elif 'DtoH' in func:
                        self.DtoHTime += pct
                    elif 'memset' in func:
                        self.MemsetTime += pct
                    else: # all others are assumed to be API calls
                        self.APITime += pct
                else: # I guess we assume these are kernel calls
                    self.KernelTime += pct

            # Print here, handle elsewhere
            if self.Total() == 0:
                print('error opening profile data file ' + OutFileName)                

    # Value constructor
        elif ctorType is 'val':
            (k, m, d, h, a) = args
            self.KernelTime = float(k)
            self.MemsetTime = float(m)
            self.DtoHTime = float(d)
            self.HtoDTime= float(h)
            self.APITime = float(a)
        
    # add two together, helps to average
    def __add__(A, B):
        k = A.KernelTime + B.KernelTime
        m = A.MemsetTime + B.MemsetTime
        d = A.DtoHTime + B.DtoHTime
        h = A.HtoDTime + B.HtoDTime
        a = A.APITime + B.APITime

        return ProfileData(k, m, d, h, a)
    
    # Needed for sum
    def __radd__(self, other):
        return self + other
        
    # Scalar division, for normalization
    def __truediv__(A, s):
        ret = A
        s = float(s)
        ret.KernelTime /= s
        ret.MemsetTime /= s
        ret.DtoHTime /= s
        ret.HtoDTime /= s
        ret.APITime /= s
        return ret
    
    def Total(self):
        return sum(self.AsList())

    # Useful for numpy array constructor
    def AsList(self):
        return [self.KernelTime, self.MemsetTime, self.DtoHTime, self.HtoDTime, self.APITime]
        
class Profiler:
    def __init__(self, progName, pSize, nIt, pattern):
        self.ProgName = progName
        self.ProbSize = pSize
        self.NumIt = nIt
        self.Pattern = pattern
        
        # maintain an average here
        self.__accum = ProfileData(0,0,0,0,0)
        self.__counter = 0
    def Execute(self, ExeName):
        # nvprof will make this output file
        profDataFile = self.ProgName + '_' + str(self.ProbSize) + '.prof'
        
        if ExeName is not 'None':
            if (system() == 'Windows'):
                ExeName += '.exe'

            # format nvprof command
            nvprofCmd = ['nvprof']
            nvprofCmd += ['--profile-api-trace', 'none']
            nvprofCmd += ['--log-file', profDataFile]
            nvprofCmd += ['--profile-from-start-off']
            nvprofCmd += [ExeName]
            nvprofCmd += [self.ProgName]
            nvprofCmd += ['profile']
            nvprofCmd += [self.ProbSize]
            nvprofCmd += [1]
            nvprofCmd += [self.NumIt]
            nvprofCmd += [self.Pattern]
            nvprofCmd = [str(x) for x in nvprofCmd]
        
            # run nvprof, get data, add to list
            ret = sp.call(nvprofCmd, shell = (system() == 'Windows'))
            if (ret != 0):
                print('Error profiling call: ') + str(bmCmd)
                return None
            else:
                print('Successfully profiled program {}, N = {}'.format(self.ProgName, self.ProbSize))
        
            
        # Add to averager and return data object
        pd = ProfileData(profDataFile)

        # see if we got any actual data
        if (pd.Total() == 0):
            return None
        
        self.__accum += pd
        self.__counter += 1
        return pd
        
    def GetAveragedData(self):
        return self.__accum / self.__counter
