#include <assert.h>
#include <vector>
#include "Scale_Forward.h"

__global__ void scale_forward(int n, int c, int h, int w, 
        const float *input, float *output) {
    // Macros for accessing flattened matrices
    #define input(nn, cc)  input[(nn) * c + (cc)]
    #define output(nn, cc, hh, ww)  output[(((nn) * c + (cc)) * h + (hh)) * w + (ww)]

    const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const long parallel_size = blockDim.x * gridDim.x;
    const long N = (long) n * c * h * w;

    for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
        long idx = n;
        const long i1 = ((idx /= 1) % n);
        const long i2 = ((idx /= n) % c);
        const long i3 = ((idx /= c) % h);
        const long i4 = ((idx /= h) % w);
        output(i1, i2, i3, i4) = input(i1, i2);
    }
}

Scale_Forward::Scale_Forward(cudnnHandle_t handle, vector<double> &args, bool is_UVM) : 
        handle(handle), is_UVM(is_UVM) {
    // 0. batch_size    1. in_channels   2. output_height    3. output_width
    output_n   = args[0];   output_c   = args[1];   output_h   = args[2];   output_w   = args[3];
    input_ratio = args[4]; output_ratio = args[5];

    if (!is_UVM) {
        CUDA_CALL(cudaMalloc(&input_data, (long) output_n * output_c * sizeof(float)));
        CUDA_CALL(cudaMalloc(&output_data, (long) output_n * output_c * output_h * output_w * sizeof(float)));
        GPUFillRand(input_data, (long) output_n * output_c * sizeof(float));
    }
    cudaDeviceSynchronize();
}

Scale_Forward::~Scale_Forward() {
    if (!is_UVM) {
        CUDA_CALL(cudaFree(input_data));
        CUDA_CALL(cudaFree(output_data));
    }
}

float Scale_Forward::Run() {
    if (is_UVM) {
        CUDA_CALL(cudaMallocManaged(&input_data, (long) output_n * output_c * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&output_data, (long) output_n * output_c * output_h * output_w * sizeof(float)));
        CPUFillRand(input_data, (long) output_n * output_c * sizeof(float));
        GPUFillRand(input_data, (long) output_n * output_c * sizeof(float) * input_ratio);
        GPUFillRand(output_data, (long) output_n * output_c * output_h * output_w * sizeof(float) * output_ratio);
        cudaDeviceSynchronize();
    }

    float milliseconds = 0;
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));

    CUDA_CALL(cudaEventRecord(start));
    scale_forward<<<output_n, output_c>>>(output_n, output_c, output_h, output_w, input_data, output_data);
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    
    if (is_UVM) {
        CUDA_CALL(cudaFree(input_data));
        CUDA_CALL(cudaFree(output_data));
    }
    return milliseconds;
}
