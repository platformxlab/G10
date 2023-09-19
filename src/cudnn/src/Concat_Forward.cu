#include <assert.h>
#include <vector>
#include "Concat_Forward.h"

__global__ void concat_forward(long n, long out_c, long h, long w, long num_input, long *c,
        float **input, float *output) {
    const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const long parallel_size = blockDim.x * gridDim.x;

    int input_idx = 0;
    long current_tail = 0;
    if (thread_pos >= out_c) return;
    while (current_tail + c[input_idx] < thread_pos) {
        current_tail += c[input_idx++];
    }
    for (long n = 0; n < out_c; n += parallel_size) {
            long idx = n + thread_pos;
            while (current_tail + c[input_idx] < idx) {
                current_tail += c[input_idx++];
            }
            output[idx] = input[input_idx][idx - current_tail];
    }
}

Concat_Forward::Concat_Forward(cudnnHandle_t handle, vector<double> &args, bool is_UVM) : 
        handle(handle), is_UVM(is_UVM) {
    // 0. batch_size    1. input_height     2. input_width
    // 3. in_channel_N  4 - N+3 in_channel
    input_n    = args[0];   input_h    = args[1];   input_w    = args[2];
    num_input  = args[3];
    output_c = 0;
    input_cs = (long *) malloc((long) num_input * sizeof(long));
    CUDA_CALL(cudaMalloc(&device_input_cs, (long) num_input * sizeof(long)));
    input_data = (float **) malloc(num_input * sizeof(float *));
    CUDA_CALL(cudaMalloc(&device_input_data, (long) num_input * sizeof(long)));
    for (long i = 0; i < num_input; i++) {
        input_cs[i] = args[4 + i];
        output_c += args[4 + i];
    }
    input_ratio = args[4+ num_input]; output_ratio = args[5+num_input];
    CUDA_CALL(cudaMemcpy(device_input_cs, input_cs, (long) num_input * sizeof(long), cudaMemcpyHostToDevice));
    
    // Alloc
    if (!is_UVM) {
        for (int i = 0; i < num_input; i++) {
            CUDA_CALL(cudaMalloc(&input_data[i], (long) input_n * input_cs[i] * input_h * input_w * sizeof(float)));
            GPUFillRand(input_data[i], (long) input_n * input_cs[i] * input_h * input_w * sizeof(float));
        }
        CUDA_CALL(cudaMalloc(&output_data, (long) input_n * output_c * input_h * input_w * sizeof(float)));
        CUDA_CALL(cudaMemcpy(device_input_data, input_data, (long) num_input * sizeof(float *), cudaMemcpyHostToDevice));
    }
    cudaDeviceSynchronize();
}

Concat_Forward::~Concat_Forward() {
    if (!is_UVM) {
        for (int i = 0; i < num_input; i++)
            CUDA_CALL(cudaFree(input_data[i]));
        CUDA_CALL(cudaFree(output_data));
    }
}

float Concat_Forward::Run() {
    if (is_UVM) {
        for (int i = 0; i < num_input; i++) {
            CUDA_CALL(cudaMallocManaged(&input_data[i], (long) input_n * input_cs[i] * input_h * input_w * sizeof(float)));
            CPUFillRand(input_data[i], (long) input_n * input_cs[i] * input_h * input_w * sizeof(float));
            GPUFillRand(input_data[i], (long) input_n * input_cs[i] * input_h * input_w * sizeof(float) * input_ratio);
        }
        CUDA_CALL(cudaMallocManaged(&output_data, (long) input_n * output_c * input_h * input_w * sizeof(float)));
        CUDA_CALL(cudaMemcpy(device_input_data, input_data, (long) num_input * sizeof(float *), cudaMemcpyHostToDevice));
        GPUFillRand(output_data, (long) input_n * output_c * input_h * input_w * sizeof(float) * output_ratio);
        cudaDeviceSynchronize();
    }

    float milliseconds = 0;
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));

    CUDA_CALL(cudaEventRecord(start));
    concat_forward<<<input_n, ceil((double) output_c / input_n)>>>(input_n, output_c, input_h, input_w, num_input, device_input_cs, device_input_data, output_data);
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    
    if (is_UVM) {
        for (int i = 0; i < num_input; i++)
            CUDA_CALL(cudaFree(input_data[i]));
        CUDA_CALL(cudaFree(output_data));
    }
    return milliseconds;
}
