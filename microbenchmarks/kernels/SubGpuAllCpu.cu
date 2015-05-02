#include "util.h"

__global__
void subset_G(float * data, int N, int3 subset){
   int idx = threadIdx.x+blockDim.x*blockIdx.x;
   if (idx < N && contains(subset, idx)) data[idx]++;
}

int main(int argc, char ** argv){
	if (argc < 3)
		return -1;

	int N = atoi(argv[1]);
	int nIt = atoi(argv[2]);
   int size = sizeof(float)*N;
   int nT(1024), nB(N/1024+1);
	if (N < 0 ||nIt < 0)
		return -2;

   srand(1);

   int3 subset;
   subset.x = (int)(((float)rand()/(float)RAND_MAX) * N);
   subset.y = (int)(((float)rand()/(float)RAND_MAX) * N);
   subset.z = (int)(((float)rand()/(float)RAND_MAX) * N);

#ifdef UMA
	float * data(0);
   cudaMallocManaged((void **)&data, size);

   for (int i=0; i<nIt; i++){
      subset_G<<<nB, nT>>>(data, N, subset);
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
      subset_G<<<nB, nT>>>(d_Data, N, subset);
      cudaMemcpy(h_Data, d_Data, size, cudaMemcpyDeviceToHost);
      for (int j=0; j<N; j++)
         h_Data[j]++;
   }

   free(h_Data);
   cudaFree(d_Data);
#endif
	return 0;
}
