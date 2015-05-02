template <typename T>
T min(T a, T b){
   return (a < b ? a : b);
}
template <typename T>
T max(T a, T b){
   return (a > b ? a : b);
}
template <typename T>
T min(T a, T m, T M){
   return min(max(a,m),M);
}

__inline__ __host__ __device__
bool contains(int3 s, int e){
   return (s.x==e || s.y==e || s.z==e);
}

__global__
void inc(float * data, int N){
   int idx = threadIdx.x+blockDim.x*blockIdx.x;
   if (idx < N) data[idx]++;
}
