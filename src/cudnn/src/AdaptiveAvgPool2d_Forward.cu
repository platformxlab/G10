#include <assert.h>
#include <vector>
#include "AdaptiveAvgPool2d_Forward.h"

AdaptiveAvgPool2d_Forward::AdaptiveAvgPool2d_Forward(cudnnHandle_t handle, vector<double> &args, bool is_UVM) : 
        handle(handle), is_UVM(is_UVM) {
    // 0. batch_size    1. in_channels   2. input_height  3. input_width
    // 4. kernel_size_h 5. kernel_size_w
    // 6. padding_h     7. padding_w     8. stride_h      9. stride_w
    input_n    = args[0];   input_c    = args[1];   input_h    = args[2];   input_w    = args[3];
    k_size_h   = args[4];   k_size_w   = args[5];   
    padding_h  = args[6];   padding_w  = args[7];   stride_h   = args[8];   stride_w   = args[9]; 
    input_ratio = args[10]; output_ratio = args[11];

    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
    CUDNN_CALL(cudnnCreatePoolingDescriptor(&avgpool_descriptor));
    CUDNN_CALL(cudnnSetPooling2dDescriptor(
            avgpool_descriptor,
            CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
            CUDNN_NOT_PROPAGATE_NAN,
            k_size_h, k_size_w,
            padding_h, padding_w,
            stride_h, stride_w));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&output_descriptor));

    // SetInputDescriptor
    CUDNN_CALL(cudnnSetTensor4dDescriptor(
            input_descriptor,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            input_n, input_c, input_h, input_w));
    // SetOutputDescriptor
    CUDNN_CALL(cudnnGetPooling2dForwardOutputDim(
            avgpool_descriptor, 
            input_descriptor,
            &output_n, &output_c, &output_h, &output_w));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(
            output_descriptor,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            output_n, output_c, output_h, output_w));
    // Alloc
    if (!is_UVM) {
        CUDA_CALL(cudaMalloc(&input_data, (long) input_n * input_c * input_h * input_w * sizeof(float)));
        CUDA_CALL(cudaMalloc(&output_data, (long) output_n * output_c * output_h * output_w * sizeof(float)));
        GPUFillRand(input_data, (long) input_n * input_c * input_h * input_w * sizeof(float));
    }
    cudaDeviceSynchronize();
}

AdaptiveAvgPool2d_Forward::~AdaptiveAvgPool2d_Forward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_descriptor));
    CUDNN_CALL(cudnnDestroyPoolingDescriptor(avgpool_descriptor));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(output_descriptor));
    if (!is_UVM) {
        CUDA_CALL(cudaFree(input_data));
        CUDA_CALL(cudaFree(output_data));
    }
}

float AdaptiveAvgPool2d_Forward::Run() {
    if (is_UVM) {
        CUDA_CALL(cudaMallocManaged(&input_data, (long) input_n * input_c * input_h * input_w * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&output_data, (long) output_n * output_c * output_h * output_w * sizeof(float)));
        CPUFillRand(input_data, (long) input_n * input_c * input_h * input_w * sizeof(float));
        GPUFillRand(input_data, (long) input_n * input_c * input_h * input_w * sizeof(float) * input_ratio);
        GPUFillRand(output_data, (long) output_n * output_c * output_h * output_w * sizeof(float) * output_ratio);
        cudaDeviceSynchronize();
    }

    float milliseconds = 0;
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));

    CUDA_CALL(cudaEventRecord(start));
    CUDNN_CALL(cudnnPoolingForward(
            handle,
            avgpool_descriptor,
            &alpha,
            input_descriptor, input_data,
            &beta,
            output_descriptor, output_data));
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    
    if (is_UVM) {
        CUDA_CALL(cudaFree(input_data));
        CUDA_CALL(cudaFree(output_data));
    }
    return milliseconds;
}
