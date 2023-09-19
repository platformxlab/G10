#include "../include/cudadnnUtil.cuh"


__global__ void forwardPass_Dropout(float* input, float* output, float* musk, int N){

    const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int parallel_size = blockDim.x * gridDim.x;

    for(int n = 0; n < N; n += parallel_size){
        int idx = n + thread_pos;
        if(idx < N) {
            output[idx] = (musk[idx] == 0.0f) ? 0 : input[idx];
        }
    }
}


__global__ void backwardPass_Dropout(float* d_output, float* d_input, float* musk, int N){

    const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int parallel_size = blockDim.x * gridDim.x;

    for(int n = 0; n < N; n += parallel_size){
        int idx = n + thread_pos;
        if(idx < N) {
            d_input[idx] = (musk[idx] == 0.0f) ? 0 : d_output[idx];
        }
    }
}


__global__ void makeError(float *err, float *output, unsigned int Y, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
        if(idx < N) {
		    err[idx] = ((Y == idx ? 1.0f : 0.0f) - output[idx]);
        }
	}
}



__global__ void apply_grad(float *output, float *grad, const int N)
{
	const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int parallel_size = blockDim.x * gridDim.x;

    for(int n = 0; n < N; n += parallel_size){
        int idx = n + thread_pos;
        if(idx < N) {
            output[idx] += dt * grad[idx];
        }
    }
}


__global__ void transfer_A0(float* data, float* activation_tensor, const int N){
    const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int parallel_size = blockDim.x * gridDim.x;

    for(int n = 0; n < N; n += parallel_size){
        int idx = n + thread_pos;
        if(idx < N) {
            activation_tensor[idx] = data[idx];
        }
    }
}