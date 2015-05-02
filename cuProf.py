import sys
from subprocess import Popen, PIPE, call

import numpy as np
import matplotlib.pyplot as plt

#rather than read from args, just hard code this at some point

#args = sys.argv[1:len(sys.argv)]

N = str(1024)
nIt = str(1000)
dim = str(1)

programs = []

'''
exe_dir = './symlink/'
programs.append([exe_dir+'gsRelax', N, nIt, dim])
programs.append([exe_dir+'AllGpuAllCpu', N, nIt])
programs.append([exe_dir+'AllGpuSubCpu', N, nIt])
programs.append([exe_dir+'SubGpuAllCpu', N, nIt])
programs.append([exe_dir+'SubGpuAllCpu_r', N, nIt])
'''
exe_dir = './symlink/rodinia/'
rod_data_dir = 'rodinia_3.0/data'
programs.append([exe_dir+'backprop 65536'])
programs.append([exe_dir+'bfs '+rod_data_dir+'/bfs/graph1MW_6.txt'])
#programs.append([exe_dir+'dwt2d '+exe_dir+'dwt2d/192.bmp -d 192x192 -f -5 -l 3'])
programs.append([exe_dir+'euler3d '+rod_data_dir+'/cfd/fvcorr.domn.097K'])
programs.append([exe_dir+'gaussian -f '+rod_data_dir+'/gaussian/matrix4.txt'])
programs.append([exe_dir+'heartwall '+rod_data_dir+'/heartwall/test.avi 20'])
programs.append([exe_dir+'hotspot 512 2 2 '+rod_data_dir+'/hotspot/temp_512 '+rod_data_dir+'/hotspot/power_512 output.out'])
programs.append([exe_dir+'hybridsort r'])
programs.append([exe_dir+'kmeans -o -i '+rod_data_dir+'/kmeans/kdd_cup'])
programs.append([exe_dir+'lavaMD -boxes1d 10'])
programs.append([exe_dir+'leukocyte '+rod_data_dir+'/leukocyte/testfile.avi 5'])
programs.append([exe_dir+'lud_cuda -s 256 -v'])
programs.append([exe_dir+'needle 2048 10'])
#programs.append([exe_dir+'nn filelist_4 -r 5 -lat 30 -lng 90'])
programs.append([exe_dir+'particlefilter_float -x 128 -y 128 -z 10 -np 1000'])
programs.append([exe_dir+'particlefilter_naive -x 128 -y 128 -z 10 -np 1000'])
programs.append([exe_dir+'pathfinder 100000 100 20'])
programs.append([exe_dir+'sc_gpu 10 20 256 65536 65536 1000 none output.txt 1'])
programs.append([exe_dir+'srad_v1 100 0.5 502 458'])
programs.append([exe_dir+'srad_v2 2048 2048 0 127 0 127 0.5 2'])

kernel = []
DtoH = []
HtoD = []
API = []
names = []

for a in programs:
	n = a[0].split('/')[3].split()[0]
	print n+'\n'
	cmd = 'nvprof --profile-api-trace none --log-file output/'+n+'.dat '+' '.join(a)
	print cmd+'\n\n'
	p = Popen(cmd, shell=True, stdin=PIPE,stdout=PIPE, stderr=PIPE)
	out,err = p.communicate()
	names.append(n) #check ret first
	f = open('output/'+n+'.dat', 'r')
	for i in range (0,4):
		f.readline()
	dth = 0.
	htd = 0.
	api = 0.
	k   = 0.
	for line in f:
		if line[0] == '=' or len(line) == 1:
			break
		data = line.split()
		pct = float(data[0][:-1])
		name = ''.join(data[6:])
		if (name.find('CUDA') >= 0 and name.find('[') >= 0 and name.find(']') >= 0):
			if (name.find('HtoD') >= 0):
				print name + ' has htod in it '+str(pct)
				htd += pct
			elif (name.find('DtoH') >= 0):
				print name + ' has dtoh in it '+str(pct)
				dth += pct
			else:
				print name + ' has apic in it '+str(pct)
				api += pct
		else:
			print name + ' has kernel in it '+str(pct)
			k += pct
	HtoD.append(htd)
	DtoH.append(dth)
	API.append(api)
	kernel.append(k)
	f.close()

ind = np.arange(len(names))
width=0.3

b1 = np.add(np.float32(HtoD),np.float32(DtoH))
b2 = np.add(np.float32(b1), np.float32(API))

p1 = plt.bar(ind, HtoD, width, color='#ff0000')
p2 = plt.bar(ind, DtoH, width, bottom=HtoD, color='#00ff00')
p3 = plt.bar(ind, API, width, bottom=b1, color='#0000ff')
p4 = plt.bar(ind, kernel, width, bottom=b2, color='#f0f0f0')

plt.xticks(ind + width/2., names )
plt.yticks(np.arange(0,135,10))
plt.legend( (p1[0], p2[0], p3[0], p4[0]), ('HtoD', 'DtoH', 'API', 'kernel') )

print HtoD
print DtoH
print API
print kernel

plt.show()
