#include <assert.h>
#include <vector>
#include "Scale_Backward.h"

__global__ void scale_backward(int n, int c, int h, int w, 
        const float *d_input, float *d_output) {
    // Macros for accessing flattened matrices
    #define d_input(nn, cc)  d_input[(nn) * c + (cc)]
    #define d_output(nn, cc, hh, ww)  d_output[(((nn) * c + (cc)) * h + (hh)) * w + (ww)]

    const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const long parallel_size = blockDim.x * gridDim.x;
    const long N = (long) n * c * h * w;

    for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
        long idx = n;
        const long i1 = ((idx /= 1) % n);
        const long i2 = ((idx /= n) % c);
        const long i3 = ((idx /= c) % h);
        const long i4 = ((idx /= h) % w);
        d_output(i1, i2, i3, i4) = d_input(i1, i2);
    }
}

Scale_Backward::Scale_Backward(cudnnHandle_t handle, vector<double> &args, bool is_UVM) : 
        handle(handle), is_UVM(is_UVM) {
    // 0. batch_size    1. in_channels   2. output_height    3. output_width
    output_n   = args[0];   output_c   = args[1];   output_h   = args[2];   output_w   = args[3];
    input_ratio = args[4]; output_ratio = args[5];

    if (!is_UVM) {
        CUDA_CALL(cudaMalloc(&d_input_data, (long) output_n * output_c * sizeof(float)));
        CUDA_CALL(cudaMalloc(&d_output_data, (long) output_n * output_c * output_h * output_w * sizeof(float)));
        GPUFillRand(d_output_data, (long) output_n * output_c * output_h * output_w * sizeof(float));
    }
    cudaDeviceSynchronize();
}

Scale_Backward::~Scale_Backward() {
    if (!is_UVM) {
        CUDA_CALL(cudaFree(d_input_data));
        CUDA_CALL(cudaFree(d_output_data));
    }
}

float Scale_Backward::Run() {
    if (is_UVM) {
        CUDA_CALL(cudaMallocManaged(&d_input_data, (long) output_n * output_c * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&d_output_data, (long) output_n * output_c * output_h * output_w * sizeof(float)));
        CPUFillRand(d_output_data, (long) output_n * output_c * output_h * output_w * sizeof(float));
        GPUFillRand(d_output_data, (long) output_n * output_c * output_h * output_w * sizeof(float) * input_ratio);
        GPUFillRand(d_input_data, (long) output_n * output_c * sizeof(float) * output_ratio);
        cudaDeviceSynchronize();
    }

    float milliseconds = 0;
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));

    CUDA_CALL(cudaEventRecord(start));
    scale_backward<<<output_n, output_c>>>(output_n, output_c, output_h, output_w, d_input_data, d_output_data);
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    
    if (is_UVM) {
        CUDA_CALL(cudaFree(d_input_data));
        CUDA_CALL(cudaFree(d_output_data));
    }
    return milliseconds;
}
