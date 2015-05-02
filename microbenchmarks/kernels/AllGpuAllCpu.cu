#include "util.h"

int main(int argc, char ** argv){
	if (argc < 3)
		return -1;
	int N = atoi(argv[1]);
	int nIt = atoi(argv[2]);
   int nT(1024), nB(N/1024+1);
   int size = sizeof(float)*N;

	if (N < 0 || nIt < 0)
		return -1;

#ifdef UMA
   float * data(0);
	cudaMallocManaged((void **)&data, size);
   for (int i=0; i<nIt; i++){
      inc<<<nB, nT>>>(data, N);
		cudaDeviceSynchronize();      
      for (int j=0; j<N; j++)
			data[j]++;
   }

   free(data);
#else
   float * h_Data(0), * d_Data(0);
   h_Data = (float *)malloc(size);
   cudaMalloc((void **)&d_Data, size);
   for (int i=0; i<nIt; i++){
      cudaMemcpy(d_Data, h_Data, size, cudaMemcpyHostToDevice);
      inc<<<nB, nT>>>(d_Data, N);
      cudaMemcpy(h_Data, d_Data, size, cudaMemcpyDeviceToHost);
      for (int j=0; j<N; j++)
			h_Data[j]++;
   }

   free(h_Data);
   cudaFree(d_Data);
#endif
	return 0;
}
