#include <assert.h>
#include <vector>
#include "BatchNorm2d_Forward.h"

BatchNorm2d_Forward::BatchNorm2d_Forward(cudnnHandle_t handle, vector<double> &args, bool is_UVM) : 
        handle(handle), is_UVM(is_UVM) {
    // 0. batch_size    1. in_channels   2. input_height    3. input_width
    input_n    = args[0];   input_c    = args[1];   input_h    = args[2];   input_w  = args[3];
    input_ratio = args[4]; output_ratio = args[5];

    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&batch_norm_descriptor));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&output_descriptor));

    // SetInputDescriptor
    CUDNN_CALL(cudnnSetTensor4dDescriptor(
            input_descriptor,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            input_n, input_c, input_h, input_w));
    // SetBatchNormDescriptor
    CUDNN_CALL(cudnnSetTensor4dDescriptor(
            batch_norm_descriptor,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            1, input_c, 1, 1));
    // SetOutputDescriptor
    CUDNN_CALL(cudnnSetTensor4dDescriptor(
            output_descriptor,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            input_n, input_c, input_h, input_w));
    // Alloc
    if (!is_UVM) {
        CUDA_CALL(cudaMalloc(&input_data, (long) input_n * input_c * input_h * input_w * sizeof(float)));
        CUDA_CALL(cudaMalloc(&result_running_mean, (long) input_n * input_c * input_h * input_w * sizeof(float)));
        CUDA_CALL(cudaMalloc(&result_running_variance, (long) input_n * input_c * input_h * input_w * sizeof(float)));
        CUDA_CALL(cudaMalloc(&output_data, (long) input_n * input_c * input_h * input_w * sizeof(float)));
        CUDA_CALL(cudaMalloc(&bn_scale, (long) input_c * sizeof(float)));
        CUDA_CALL(cudaMalloc(&bn_bias, (long) input_c * sizeof(float)));
        GPUFillRand(input_data, (long) input_n * input_c * input_h * input_w * sizeof(float));
        GPUFillRand(result_running_mean, (long) input_n * input_c * input_h * input_w * sizeof(float));
        GPUFillRand(result_running_variance, (long) input_n * input_c * input_h * input_w * sizeof(float));
        GPUFillRand(bn_scale, (long) input_c * sizeof(float));
        GPUFillRand(bn_bias, (long) input_c * sizeof(float));
    }
    cudaDeviceSynchronize();
}

BatchNorm2d_Forward::~BatchNorm2d_Forward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_descriptor));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(batch_norm_descriptor));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(output_descriptor));
    if (!is_UVM) {
        CUDA_CALL(cudaFree(input_data));
        CUDA_CALL(cudaFree(output_data));
        CUDA_CALL(cudaFree(result_running_mean));
        CUDA_CALL(cudaFree(result_running_variance));
        CUDA_CALL(cudaFree(bn_scale));
        CUDA_CALL(cudaFree(bn_bias));
    }
}

float BatchNorm2d_Forward::Run() {
    if (is_UVM) {
        CUDA_CALL(cudaMallocManaged(&input_data, (long) input_n * input_c * input_h * input_w * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&result_running_mean, (long) input_n * input_c * input_h * input_w * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&result_running_variance, (long) input_n * input_c * input_h * input_w * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&output_data, (long) input_n * input_c * input_h * input_w * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&bn_scale, (long) input_c * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&bn_bias, (long) input_c * sizeof(float)));
        CPUFillRand(input_data, (long) input_n * input_c * input_h * input_w * sizeof(float));
        CPUFillRand(result_running_mean, (long) input_n * input_c * input_h * input_w * sizeof(float));
        CPUFillRand(result_running_variance, (long) input_n * input_c * input_h * input_w * sizeof(float));
        CPUFillRand(bn_scale, (long) input_c * sizeof(float));
        CPUFillRand(bn_bias, (long) input_c * sizeof(float));

        GPUFillRand(input_data, (long) input_n * input_c * input_h * input_w * sizeof(float) * input_ratio);
        GPUFillRand(result_running_mean, (long) input_n * input_c * input_h * input_w * sizeof(float) * input_ratio);
        GPUFillRand(result_running_variance, (long) input_n * input_c * input_h * input_w * sizeof(float) * input_ratio);
        GPUFillRand(output_data, (long) input_n * input_c * input_h * input_w * sizeof(float) * output_ratio);
        GPUFillRand(bn_scale, (long) input_c * sizeof(float) * input_ratio);
        GPUFillRand(bn_bias, (long) input_c * sizeof(float) * input_ratio);
        cudaDeviceSynchronize();
    }

    float one = 1;
    float zero = 0;
    
    float milliseconds = 0;
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));

    CUDA_CALL(cudaEventRecord(start));
    CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
        handle,
        CUDNN_BATCHNORM_SPATIAL,/* cudnnBatchNormMode_t mode */
        &one,
        &zero,
        input_descriptor,       /* const cudnnTensorDescriptor_t xDesc */
        input_data,             /* const void *x */
        output_descriptor,      /* const cudnnTensorDescriptor_t yDesc */
        output_data,            /* void *y */
        batch_norm_descriptor,  /* const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc */
        bn_scale,
        bn_bias,
        exponentialAverageFactor,
        result_running_mean,
        result_running_variance,
        epsilon,
        nullptr,
        nullptr));
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    
    if (is_UVM) {
        CUDA_CALL(cudaFree(input_data));
        CUDA_CALL(cudaFree(output_data));
        CUDA_CALL(cudaFree(result_running_mean));
        CUDA_CALL(cudaFree(result_running_variance));
        CUDA_CALL(cudaFree(bn_scale));
        CUDA_CALL(cudaFree(bn_bias));
    }
    return milliseconds;
}
