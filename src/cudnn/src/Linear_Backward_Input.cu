#include <assert.h>
#include <vector>
#include "Linear_Backward_Input.h"

#define TILE_SZ_A 128
#define TILE_SZ_B 16
#define TILE_SZ_RATIO (TILE_SZ_A/TILE_SZ_B)

__global__ void backward_input(int m, int n, int k, 
        const float *input, const float *weight, float* output) {

    // Macros for accessing flattened matrices
    #define input(row, col)  input[(row) + (col) * m]
    #define weight(row, col) weight[(row) * n + (col)]
    #define output(row, col) output[(row) + (col) * m]

    __shared__ float B_shared[TILE_SZ_RATIO][TILE_SZ_B];
    int row = blockIdx.x * TILE_SZ_A + threadIdx.x;

    int n_iter_num = ceil(n * 1.0 / TILE_SZ_B);
    int k_iter_num = ceil(k * 1.0 / TILE_SZ_RATIO);
    for (int n_iter = 0; n_iter < n_iter_num; n_iter++) {
        for (int k_iter = 0; k_iter < k_iter_num; k_iter++) {
        // load weight tile into shared memory, weight is transposed
        int shared_start_row = k_iter * TILE_SZ_RATIO;
        int shared_start_col = n_iter * TILE_SZ_B;
        int shared_row_offset = threadIdx.x / TILE_SZ_B;
        int shared_col_offset = threadIdx.x % TILE_SZ_B;

        if (shared_start_row + shared_row_offset < k && shared_start_col + shared_col_offset < n) {
            B_shared[shared_row_offset][shared_col_offset] = 
                    weight(shared_start_row + shared_row_offset, shared_start_col + shared_col_offset);
        } else {
            B_shared[shared_row_offset][shared_col_offset] = 0;
        }

        __syncthreads();

        for (int j = 0; j < TILE_SZ_B; j++) {
            float output_cumulative = 0;
            for (int i = 0; i < TILE_SZ_RATIO; i++) {
                if (row < m && shared_start_col + j < n) {
                    output_cumulative += input(row, k_iter * TILE_SZ_RATIO + i) * B_shared[i][j];
                }
            }
            output(row, shared_start_col + j) += output_cumulative;
        }
        
        __syncthreads();
        }
    }
}

Linear_Backward_Input::Linear_Backward_Input(cudnnHandle_t handle, vector<double> &args, bool is_UVM) : 
        handle(handle), is_UVM(is_UVM) {
    // 0. batch_size    1. in_channels   2. input_height    3. input_width
    // 4. h_in          5. h_out
    input_n    = args[0];   input_c    = args[1];   input_h    = args[2];   input_w  = args[3];
    h_in       = args[4];   h_out      = args[5];
    reshape = (long) input_n * input_c * input_h * input_w / h_in;
    input_ratio = args[6]; output_ratio = args[7];

    if (!is_UVM) {
        CUDA_CALL(cudaMalloc(&d_input_data, (long) reshape * h_in * sizeof(float)));
        CUDA_CALL(cudaMalloc(&weight_data, (long) h_in * h_out * sizeof(float)));
        CUDA_CALL(cudaMalloc(&d_output_data, (long) reshape * h_out * sizeof(float)));
        GPUFillRand(weight_data, (long) h_in * h_out * sizeof(float));
        GPUFillRand(d_output_data, (long) reshape * h_out * sizeof(float));
    }
    cudaDeviceSynchronize();
}

Linear_Backward_Input::~Linear_Backward_Input() {
    if (!is_UVM) {
        CUDA_CALL(cudaFree(d_input_data));
        CUDA_CALL(cudaFree(weight_data));
        CUDA_CALL(cudaFree(d_output_data));
    }
}

float Linear_Backward_Input::Run() {
    if (is_UVM) {
        CUDA_CALL(cudaMallocManaged(&d_input_data, (long) reshape * h_in * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&weight_data, (long) h_in * h_out * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&d_output_data, (long) reshape * h_out * sizeof(float)));
        CPUFillRand(weight_data, (long) h_in * h_out * sizeof(float));
        CPUFillRand(d_output_data, (long) reshape * h_out * sizeof(float));

        GPUFillRand(weight_data, (long) h_in * h_out * sizeof(float) * input_ratio);
        GPUFillRand(d_output_data, (long) reshape * h_out * sizeof(float) * input_ratio);
        GPUFillRand(d_input_data, (long) reshape * h_in * sizeof(float) * output_ratio);
        cudaDeviceSynchronize();
    }

    dim3 BlockSize(TILE_SZ_A, 1, 1);
    dim3 GridSize(ceil((double) reshape / TILE_SZ_A), 1, 1);

    float milliseconds = 0;
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));

    CUDA_CALL(cudaEventRecord(start));
    backward_input<<<GridSize, BlockSize>>>(reshape, h_in, h_out, d_output_data, weight_data, d_input_data);
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    
    if (is_UVM) {
        CUDA_CALL(cudaFree(d_input_data));
        CUDA_CALL(cudaFree(weight_data));
        CUDA_CALL(cudaFree(d_output_data));
    }
    return milliseconds;
}
