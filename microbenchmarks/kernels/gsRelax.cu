#define UMA 0

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/******
	TODO:
		Handle different problem sizes more elegantly
******/

__inline__ __host__ __device__
uint32_t get2Didx(uint32_t x, uint32_t y, uint32_t N){
	return x + y * N;
}

__global__
void gsRelax_Laplacian2D(float * in, float * out, uint32_t N, uint32_t odd){
	uint32_t idx_X  = 2*(threadIdx.x+blockDim.x*blockIdx.x) + odd%2;
	uint32_t idx_Y  = threadIdx.y+blockDim.y*blockIdx.y;

	if (idx_X > 0 && idx_X < N-1 && idx_Y > 0 && idx_Y < N-1){
		uint32_t idx    = get2Didx(idx_X, idx_Y, N);
		uint32_t idx_x1 = get2Didx(idx_X-1, idx_Y, N);
		uint32_t idx_x2 = get2Didx(idx_X+1, idx_Y, N);
		uint32_t idx_y1 = get2Didx(idx_X, idx_Y-1, N);
		uint32_t idx_y2 = get2Didx(idx_X, idx_Y+1, N);

		float sum = 
			in[idx_x1]+
			in[idx_x2]+
			in[idx_y1]+
			in[idx_y2];

		out[idx] = 0.25f * sum;
	}
}

__global__
void gsRelax_Laplacian1D(float * in, float * out, uint32_t N, uint32_t odd){
	uint32_t idx = 2*(threadIdx.x+blockDim.x*blockIdx.x) + odd%2;
	
	if (idx > 0 && idx < N-1){
		float sum = in[idx-1]+in[idx+1];
		out[idx] = 0.5f * sum;
	}
}

inline float getResidueSq(float * in, float * out, uint32_t N){
	float r(0);
	for (uint32_t i=0; i<N; i++)
		r += pow(out[i]-in[i], 2);
	return r;
}

void makeData(float * data, uint32_t N){
	srand(1);//time(0));
	for (uint32_t i=0; i<N; i++)
		data[i] = (float)rand()/(float)RAND_MAX;
}

template <typename T>
inline void swap(T& a, T& b){
	T c(a);
	a = b;
	b = c;
}

void relax_UMA(uint32_t N, uint32_t dim, uint32_t nIt, float minRes){
	//Just a stupid pad
	while (N%1024) N++;
	dim = (dim%2 ? 1 : 2);
	N = (dim==1 ? N : N*N);
	uint32_t size = sizeof(float)*N;
	float * data_A(0), * data_B(0);
	float res(0);
	
	cudaMallocManaged((void **)&data_A, size);
	cudaMallocManaged((void **)&data_B, size);

	makeData(data_A, N);
	makeData(data_B, N);

	//Repetitive, I know, but I didn't want to introduce overhead during iteration
	if (dim==1){
		int nT(1024), nB((N / 1024)/2+1);
		for (int i=0; i<2*nIt/* && res > minRes*/;){
			gsRelax_Laplacian1D<<<nB, nT>>>(data_A, data_B, N, i++); 
			gsRelax_Laplacian1D<<<nB, nT>>>(data_A, data_B, N, i++); 
			cudaDeviceSynchronize();
			res = sqrt(getResidueSq(data_A, data_B, N));
		
			swap(data_A, data_B);	
		}
	}
	else if (dim==2){
		uint32_t len = (uint32_t)sqrt(N);
		dim3 nB, nT;
		nB.x = (len/32)/2+1; nB.y = (len/32)/2+1;
		nT.x = 32; nT.y = 32;
		for (int i=0; i<2*nIt/* && res > minRes*/;){
			gsRelax_Laplacian2D<<<nB, nT>>>(data_A, data_B, len, i++); 
			gsRelax_Laplacian2D<<<nB, nT>>>(data_A, data_B, len, i++); 
			cudaDeviceSynchronize();
			res = sqrt(getResidueSq(data_A, data_B, N));
		
			swap(data_A, data_B);	
		}
	}

	printf("%f\n", res);	

	free(data_A);
	free(data_B);
}

void relax(uint32_t N, uint32_t dim, uint32_t nIt, float minRes){
	//Just a stupid pad
	while (N%1024) N++;
	dim = (dim%2 ? 1 : 2);
	N = (dim==1 ? N : N*N);
	uint32_t size = sizeof(float)*N;
	float * h_Data_A(0), * d_Data_A(0), * h_Data_B(0), * d_Data_B(0);
	float res(0);
	h_Data_A = (float *)malloc(size);
	h_Data_B = (float *)malloc(size);
	cudaMalloc((void **)&d_Data_A, size); 
	cudaMalloc((void **)&d_Data_B, size); 

	makeData(h_Data_A, N);
	makeData(h_Data_B, N);

	//Repetitive, I know, but I didn't want to introduce overhead during iteration
	if (dim==1){
		int nT(1024), nB((N / 1024)/2+1);
		for (int i=0; i<2*nIt/* && res > minRes*/;){
			cudaMemcpy(d_Data_A, h_Data_A, size, cudaMemcpyHostToDevice);
			cudaMemcpy(d_Data_B, h_Data_B, size, cudaMemcpyHostToDevice);
			gsRelax_Laplacian1D<<<nB, nT>>>(d_Data_A, d_Data_B, N, i++); 
			gsRelax_Laplacian1D<<<nB, nT>>>(d_Data_A, d_Data_B, N, i++); 
			cudaMemcpy(h_Data_A, d_Data_A, size, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_Data_B, d_Data_B, size, cudaMemcpyDeviceToHost);
			res = sqrt(getResidueSq(h_Data_A, h_Data_B, N));
		
			swap(h_Data_A, h_Data_B);	
			swap(d_Data_A, d_Data_B);	
		}
	}
	else if (dim==2){
		uint32_t len = (uint32_t)sqrt(N);
		dim3 nB, nT;
		nB.x = (len/32)/2+1; nB.y = (len/32)/2+1;
		nT.x = 32; nT.y = 32;
		for (int i=0; i<2*nIt/* && res > minRes*/;){
			cudaMemcpy(d_Data_A, h_Data_A, size, cudaMemcpyHostToDevice);
			cudaMemcpy(d_Data_B, h_Data_B, size, cudaMemcpyHostToDevice);
			gsRelax_Laplacian2D<<<nB, nT>>>(d_Data_A, d_Data_B, len, i++); 
			gsRelax_Laplacian2D<<<nB, nT>>>(d_Data_A, d_Data_B, len, i++); 
			cudaMemcpy(h_Data_A, d_Data_A, size, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_Data_B, d_Data_B, size, cudaMemcpyDeviceToHost);
			res = sqrt(getResidueSq(h_Data_A, h_Data_B, N));

			swap(h_Data_A, h_Data_B);	
			swap(d_Data_A, d_Data_B);	
		}
	}

	printf("%f\n", res);	

	free(h_Data_A);
	free(h_Data_B);
	cudaFree(d_Data_A);
	cudaFree(d_Data_B);
}

int main(int argc, char ** argv){
	if (argc < 4)
		return -1;
	int N = atoi(argv[1]);
	int nIt = atoi(argv[2]);
	int dim = atoi(argv[3]);
	if (N < 0 || nIt < 0 || !(dim & 3))
		return -1;
//	const uint32_t /*N(2048), */nIt(1000), dim(2);
	const float minRes(0.1f);

	relax(N,dim,nIt,minRes);

	return 0;
}
