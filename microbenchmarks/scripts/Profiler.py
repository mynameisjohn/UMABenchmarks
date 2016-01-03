class ProfileData:
	# file constructor
	def __init__(self, fileName):
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
			if (func.find('CUDA') >= 0 and func.find('[') >= 0 and func.find(']') >= 0):
				if (func.find('HtoD')):
					self.HtoD += pct
				elif (func.find('DtoH')):
					self.DtoH += pct
				elif (func.find('memset')):
					self.MemsetTime += pct
				else: # all others are assumed to be API calls
					self.HtoD += pct
			else: # I guess we assume these are kernel calls
				self.KernelTime += pct
				
	# Value constructor
	def __init__(self, k, m, d, h, a):
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
	def __div__(A, s):
		ret = A
		s = float(s)
		ret.KernelTime /= s
		ret.MemsetTime /= s
		ret.DtoHTime /= s
		ret.HtoDTime /= s
		ret.APITime /= s
		return ret
	
	# Useful for numpy array constructor
	def AsList(self):
		return [self.KernelTime, self.MemsetTime, self.DtoHTime, self.HtoDTime, self.APITime]
		
class Profiler:
	def __init__(self, progName, pSize, nIt, pattern):
		self.ProgName = progName
		self.ProbSize = pSize
		self. NumIt = nIt
		self.Pattern = pattern
		
		# maintain an average here
		self.__accum = ProfileData(0,0,0,0,0)
		self.__counter = 0
	def Execute(self, ExeName):
		# nvprof will make this output file
		profDataFile = ProgName + '_' + str(ProbSize) + '.prof'
		
		# format nvprof command
		nvprofCmd = ['nvprof']
		nvprofCmd += ['--profile-api-trace', 'none']
		nvprofCmd += ['--log-file', profDataFile]
		nvprofCmd += ['--profile-from-start-off']
		nvprofCmd += [ExeName]
		
		# run nvprof, get data, add to list
		ret = sp.call(nvprofCmd, shell = (system() == 'Windows'))
		if (ret != 0):
			return None
			
		# Add to averager and return data object
		ret = ProfileData(profDataFile)
		
		self.__accum += ret
		self.__counter += 1
		return ret
		
	def GetAveragedData(self):
		return self.__accum / float(self.__counter)