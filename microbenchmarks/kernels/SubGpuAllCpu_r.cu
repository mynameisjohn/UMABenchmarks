__global__
void subset_G_Rand(float * in, float * out, int N, float thresh){
   int idx = threadIdx.x+blockDim.x*blockIdx.x;
   if (idx < N){
      float val = in[idx];
      if (val < thresh) out[idx] = val;
   }
}

int main(int argc, char ** argv){
	if (argc < 3)
		return -1;

	srand(1);

	int N = atoi(argv[1]);
	int nIt = atoi(argv[2]);
	float thresh = (argc < 4 ? (float)rand()/(float)RAND_MAX : atof(argv[3]));
   int nT(1024), nB(N/1024+1);
   int size = sizeof(float)*N;

	if (N < 0 || nIt < 0 || thresh < 0.f)
		return -2;

   srand(1);

   if (thresh < 0)
      thresh = (float)rand()/(float)RAND_MAX;

#ifdef UMA
	float * in(0), * out(0);
   cudaMallocManaged((void **)&in, size);
   cudaMallocManaged((void **)&out, size);

   for (int j=0; j<N; j++){
      in[j] = (float)rand()/(float)RAND_MAX;
      out[j] = (float)rand()/(float)RAND_MAX;
   }

   for (int i=0; i<nIt; i++){
      subset_G_Rand<<<nB, nT>>>(in, out, N, thresh);
		cudaDeviceSynchronize();
      for (int j=0; j<N; j++){
         in[j]++;
         out[j]++;
      }
   }

   free(in);
   free(out);
#else
	float * h_In(0), * h_Out(0), * d_In(0), * d_Out(0);
   h_In = (float *)malloc(size);
   h_Out = (float *)malloc(size);
   cudaMalloc((void **)&d_In, size);
   cudaMalloc((void **)&d_Out, size);

   for (int j=0; j<N; j++){
      h_In[j] = (float)rand()/(float)RAND_MAX;
      h_Out[j] = (float)rand()/(float)RAND_MAX;
   }

   for (int i=0; i<nIt; i++){
      cudaMemcpy(d_In, h_In, size, cudaMemcpyHostToDevice);
      cudaMemcpy(d_Out, h_Out, size, cudaMemcpyHostToDevice);
      subset_G_Rand<<<nB, nT>>>(d_In, d_Out, N, thresh);
      cudaMemcpy(h_In, d_In, size, cudaMemcpyDeviceToHost);
      cudaMemcpy(h_Out, d_Out, size, cudaMemcpyDeviceToHost);
      for (int j=0; j<N; j++){
         h_In[j]++;
         h_Out[j]++;
      }
   }

   free(h_In);
   free(h_Out);
   cudaFree(d_In);
   cudaFree(d_Out);
#endif
	return 0;
}
