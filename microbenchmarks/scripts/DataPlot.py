import matplotlib.pyplot as plt
import numpy as np
import itertools

g_ShowPlots = False

def MakeBenchmarkPlots(dBenchmarkers, filename = 'bm.pdf'):
    # Get Averaged benchmark data, make plots
    # we want normalized runtime here for each
    # problem size, for each program run
    # we have a separate list for each problem size
    BMData = dict()
    Names = None
    Indices = np.arange(len(dBenchmarkers))
    Sizes = list(dBenchmarkers.keys())
    for progs in dBenchmarkers.values():
        if Names == None:
            Names = list(progs.keys())
        for bm in progs.values():
            if bm.ProgName not in BMData.keys():
                BMData[bm.ProgName] = []
            bdAvg = bm.GetAveragedData()
            nrm = bdAvg.UMATime / bdAvg.HDTime
            BMData[bm.ProgName].append(nrm)

    marker = itertools.cycle(( '+', '.', 'o', '*'))
    for bmd, name in zip(BMData.values(), Names):
        plt.plot(Indices, bmd, label=name, marker = next(marker))

    plt.yticks(np.arange(0, 2, 0.1))
    plt.xticks(Indices, Sizes )

    plt.legend(  )

    if g_ShowPlots is True:
        plt.show()
    
    plt.savefig(filename)
    plt.clf()
    
def MakeProfilerPlot(dProfilers, filename = 'prof.pdf'):
    # get averaged data from profilers
    Kernel = np.float32([0] * len(dProfilers))
    Memset = np.float32([0] * len(dProfilers))
    DtoH = np.float32([0] * len(dProfilers))
    HtoD = np.float32([0] * len(dProfilers))
    API = np.float32([0] * len(dProfilers))
    Names = []
    Indices = []
    for prof in dProfilers.values():
        idx = len(Names)
        pdAvg = prof.GetAveragedData()
        Kernel[idx] = pdAvg.KernelTime
        Memset[idx] = pdAvg.MemsetTime
        DtoH[idx] = pdAvg.DtoHTime
        HtoD[idx] = pdAvg.HtoDTime
        API[idx] = pdAvg.APITime
        Names.append(prof.ProgName)
        Indices.append(idx)

    # convert this to a numpy array
    Indices = np.float32(Indices)

    # create "bottom" arrays
    b1 = np.add(np.float32(HtoD),np.float32(DtoH))
    b2 = np.add(np.float32(b1), np.float32(Memset))
    b3 = np.add(np.float32(b2), np.float32(API))

    # Create plots
    width = 0.3
    p1 = plt.bar(Indices, HtoD, width, color='#ff0000')
    p2 = plt.bar(Indices, DtoH, width, bottom=HtoD, color='#00ff00')
    p3 = plt.bar(Indices, Memset, width, bottom=b1, color='#0000ff')
    p4 = plt.bar(Indices, API, width, bottom=b2, color='#000000')
    p5 = plt.bar(Indices, Kernel, width, bottom=b3, color='#f0f0f0')

    plt.xticks(Indices + width / 2., Names )
    plt.yticks(np.arange(0,135,10))
    plt.legend( (p1[0], p2[0], p3[0], p4[0], p5[0]), ('HtoD', 'DtoH', 'Memset', 'API', 'Kernel') )

    if g_ShowPlots is True:
        plt.show()

    plt.savefig(filename)
    plt.clf()