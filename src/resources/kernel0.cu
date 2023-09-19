#include <cuda.h>
#include <cstdio>
#include <stdlib.h>
#include <time.h>
#include <iostream>

// Workspace
float *workspace_base;


// helper functions
void generate_data(float *base, long long size) {
    for(long long i = 0; i < size; i++){
      base[i] = (float)(rand() % 256); 
    }
}


__global__ void transfer_A0(float* data, float* activation_tensor, const int N){
    const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int parallel_size = blockDim.x * gridDim.x;

    for(int n = 0; n < N; n += parallel_size){
        int idx = n + thread_pos;
        if(idx < N){
            activation_tensor[idx] = data[idx];
        }
    }
}


float* data_base;
float* A0_base;
long size;

void run_iteration() {
    cudaEvent_t start, stop;
    float elapsedTime;
  
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    transfer_A0<<<64, 128>>>(data_base, A0_base, (int)size);
  
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("0000 %20.10f ms\n", elapsedTime);
}


// main function
int main(int argc, const  char **argv) {

    int N;
    int C;
    int H;
    int W;

    std::cin>>N>>C>>H>>W;
    size = (long)N *C * H * W;
    long size_in_byte = size*sizeof(float);

    // Workspace malloc
    cudaMallocManaged(&workspace_base, size_in_byte*2);
    
    // Weight addr assignment
    data_base = workspace_base;
    A0_base = data_base + size;

    // Data & mask generation
    generate_data(workspace_base, size*2);

    // Prefetch the data to the GPU
    int device = -1;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(A0_base, size*sizeof(float), device, NULL);

  
    // Run iteration
    run_iteration();
  
    // Free
    cudaFree(workspace_base);
  }
  