#include <assert.h>
#include <vector>
#include "Dropout_Forward.h"

Dropout_Forward::Dropout_Forward(cudnnHandle_t handle, vector<double> &args, bool is_UVM) : 
        handle(handle), is_UVM(is_UVM) {
    // 0. batch_size    1. in_channels   2. input_height    3. input_width
    input_n    = args[0];   input_c    = args[1];   input_h    = args[2];   input_w  = args[3];
    input_ratio = args[4]; output_ratio = args[5];

    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
    CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropout_descriptor));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&output_descriptor));

    // SetInputDescriptor
    CUDNN_CALL(cudnnSetTensor4dDescriptor(
            input_descriptor,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            input_n, input_c, input_h, input_w));
    // SetDropoutDescriptor
    CUDNN_CALL(cudnnDropoutGetStatesSize(
            handle, &state_size));
    CUDA_CALL(cudaMalloc(&state_data, state_size));
    CUDNN_CALL(cudnnSetDropoutDescriptor(
            dropout_descriptor,
            handle,
            dropout,
            state_data,
            state_size,
            seed));
    // SetOutputDescriptor
    CUDNN_CALL(cudnnSetTensor4dDescriptor(
            output_descriptor,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            input_n, input_c, input_h, input_w));
    // AllocMemory
    CUDNN_CALL(cudnnDropoutGetReserveSpaceSize(
            input_descriptor, &workspace_size));
    // Alloc
    if (!is_UVM) {
        CUDA_CALL(cudaMalloc(&input_data, (long) input_n * input_c * input_h * input_w * sizeof(float)));
        CUDA_CALL(cudaMalloc(&output_data, (long) input_n * input_c * input_h * input_w * sizeof(float)));
        CUDA_CALL(cudaMalloc(&workspace_data, workspace_size));
        GPUFillRand(input_data, (long) input_n * input_c * input_h * input_w * sizeof(float));
    }
    cudaDeviceSynchronize();
}

Dropout_Forward::~Dropout_Forward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_descriptor));
    CUDNN_CALL(cudnnDestroyDropoutDescriptor(dropout_descriptor));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(output_descriptor));
    if (!is_UVM) {
        CUDA_CALL(cudaFree(input_data));
        CUDA_CALL(cudaFree(output_data));
        CUDA_CALL(cudaFree(workspace_data));
        CUDA_CALL(cudaFree(state_data));
    }
}

float Dropout_Forward::Run() {
    if (is_UVM) {
        CUDA_CALL(cudaMallocManaged(&input_data, (long) input_n * input_c * input_h * input_w * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&output_data, (long) input_n * input_c * input_h * input_w * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&workspace_data, workspace_size));
        CPUFillRand(input_data, (long) input_n * input_c * input_h * input_w * sizeof(float));

        GPUFillRand(input_data, (long) input_n * input_c * input_h * input_w * sizeof(float) * input_ratio);
        GPUFillRand(output_data, (long) input_n * input_c * input_h * input_w * sizeof(float) * output_ratio);
        GPUFillRand(workspace_data, workspace_size * output_ratio);
        cudaDeviceSynchronize();
    }

    float milliseconds = 0;
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));

    CUDA_CALL(cudaEventRecord(start));
    CUDNN_CALL(cudnnDropoutForward(
            handle,
            dropout_descriptor,
            input_descriptor, input_data,
            output_descriptor, output_data,
            workspace_data, workspace_size));
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    
    if (is_UVM) {
        CUDA_CALL(cudaFree(input_data));
        CUDA_CALL(cudaFree(output_data));
        CUDA_CALL(cudaFree(workspace_data));
    }
    return milliseconds;
}

size_t Dropout_Forward::getWorkspaceSize() {
    return workspace_size;
}
