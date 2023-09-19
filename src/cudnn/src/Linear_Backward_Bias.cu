#include <assert.h>
#include <vector>
#include "Linear_Backward_Bias.h"

#define TILE_SZ_A 128
#define TILE_SZ_B 16
#define TILE_SZ_RATIO (TILE_SZ_A/TILE_SZ_B)

__global__ void backward_bias(long h_out, long reshape,
        float *d_bias, const float *d_output) {
    const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const long parallel_size = blockDim.x * gridDim.x;

    const long Z = h_out;

    for (long n = Z * thread_pos / parallel_size; 
              n < Z * (thread_pos + 1) / parallel_size; 
              n++) {
        d_bias[n] = 0;

        for (long i = 0; i < reshape; ++i) {
            d_bias[n] += d_output[i * reshape + n];
        }
    }
}

Linear_Backward_Bias::Linear_Backward_Bias(cudnnHandle_t handle, vector<double> &args, bool is_UVM) : 
        handle(handle), is_UVM(is_UVM) {
    // 0. batch_size    1. in_channels   2. input_height    3. input_width
    // 4. h_in          5. h_out
    input_n    = args[0];   input_c    = args[1];   input_h    = args[2];   input_w  = args[3];
    h_in       = args[4];   h_out      = args[5];
    input_ratio = args[6]; output_ratio = args[7];
    reshape = (long) input_n * input_c * input_h * input_w / h_in;

    if (!is_UVM) {
        CUDA_CALL(cudaMalloc(&d_bias, (long) h_out * sizeof(float)));
        CUDA_CALL(cudaMalloc(&d_output, (long) reshape * h_out * sizeof(float)));
        GPUFillRand(d_output, (long) reshape * h_out * sizeof(float));
    }
    cudaDeviceSynchronize();
}

Linear_Backward_Bias::~Linear_Backward_Bias() {
    if (!is_UVM) {
        CUDA_CALL(cudaFree(d_bias));
        CUDA_CALL(cudaFree(d_output));
    }
}

float Linear_Backward_Bias::Run() {
    if (is_UVM) {
        CUDA_CALL(cudaMallocManaged(&d_bias, (long) h_out * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&d_output, (long) reshape * h_out * sizeof(float)));
        CPUFillRand(d_output, (long) reshape * h_out * sizeof(float));

        GPUFillRand(d_output, (long) reshape * h_out * sizeof(float) * input_ratio);
        GPUFillRand(d_bias, (long) h_out * sizeof(float) * output_ratio);
        cudaDeviceSynchronize();
    }

    float milliseconds = 0;
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));

    CUDA_CALL(cudaEventRecord(start));
    backward_bias<<<input_n, input_c>>>(h_out, reshape, d_bias, d_output);
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    
    if (is_UVM) {
        CUDA_CALL(cudaFree(d_bias));
        CUDA_CALL(cudaFree(d_output));
    }
    return milliseconds;
}
