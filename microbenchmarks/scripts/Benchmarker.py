class BenchmarkData:
	def __init__(self, OutFileName):
		F = open(OutFileName, 'r')
		# File has two numbers: HD time, UMA time
		runTimes = [float(n) for n in F.readline().split('\t')]
		
		# Get time
		self.HDTime = runTimes[0]
		self.UMATime = runTimes[1]
	
	def __init__(self, hd, uma):
		self.HDTime = float(hd)
		self.UMATime = float(uma)
		
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
		# format the benchmark command
		bmCmd = [ExeName]
		bmCmd += [self.ProgName]
		bmCmd += ['benchmark']
		bmCmd += [str(ProbSize)]
		bmCmd += ['1']
		bmCmd += [str(self.NumIt)]
		bmCmd += [str(self.TestCount)]
		
		# Run the benchmarks
		ret = sp.call(TestCmd, shell = (system() == 'Windows'))
		if (ret != 0):
			return None
		
		# Get output file, return data
		outFile = self.ProgName + '_' + str(self.ProbSize) + '.txt'
		ret = BenchmarkData(outFile)
		
		# maintain accumulator here
		self.__counter += 1
		self.__accum += ret
		return ret
	
	def GetAveragedData(self):
		return self.__accum / float(self.__counter)