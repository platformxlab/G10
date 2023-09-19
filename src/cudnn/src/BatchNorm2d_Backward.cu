#include <assert.h>
#include <vector>
#include "BatchNorm2d_Backward.h"

BatchNorm2d_Backward::BatchNorm2d_Backward(cudnnHandle_t handle, vector<double> &args, bool is_UVM) : 
        handle(handle), is_UVM(is_UVM) {
    // 0. batch_size    1. in_channels   2. input_height    3. input_width
    input_n    = args[0];   input_c    = args[1];   input_h    = args[2];   input_w  = args[3];
    input_ratio = args[4]; output_ratio = args[5];

    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&batch_norm_descriptor));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&d_input_descriptor));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&d_output_descriptor));

    // SetInputDescriptor
    CUDNN_CALL(cudnnSetTensor4dDescriptor(
            input_descriptor,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            input_n, input_c, input_h, input_w));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(
            d_output_descriptor,
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
            d_input_descriptor,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            input_n, input_c, input_h, input_w));
    // Alloc
    if (!is_UVM) {
        CUDA_CALL(cudaMalloc(&input_data, (long) input_n * input_c * input_h * input_w * sizeof(float)));
        CUDA_CALL(cudaMalloc(&d_input_data, (long) input_n * input_c * input_h * input_w * sizeof(float)));
        CUDA_CALL(cudaMalloc(&d_output_data, (long) input_n * input_c * input_h * input_w * sizeof(float)));
        CUDA_CALL(cudaMalloc(&bn_scale, (long) input_c * sizeof(float)));
        CUDA_CALL(cudaMalloc(&d_bn_scale, (long) input_c * sizeof(float)));
        CUDA_CALL(cudaMalloc(&d_bn_bias, (long) input_c * sizeof(float)));
        GPUFillRand(input_data, (long) input_n * input_c * input_h * input_w * sizeof(float));
        GPUFillRand(d_output_data, (long) input_n * input_c * input_h * input_w * sizeof(float));
        GPUFillRand(bn_scale, (long) input_c * sizeof(float));
        GPUFillRand(d_bn_scale, (long) input_c * sizeof(float));
        GPUFillRand(d_bn_bias, (long) input_c * sizeof(float));
    }
    cudaDeviceSynchronize();
}

BatchNorm2d_Backward::~BatchNorm2d_Backward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_descriptor));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(batch_norm_descriptor));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(d_input_descriptor));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(d_output_descriptor));
    if (!is_UVM) {
        CUDA_CALL(cudaFree(input_data));
        CUDA_CALL(cudaFree(d_input_data));
        CUDA_CALL(cudaFree(d_output_data));
        CUDA_CALL(cudaFree(d_bn_scale));
        CUDA_CALL(cudaFree(d_bn_bias));
    }
}

float BatchNorm2d_Backward::Run() {
    if (is_UVM) {
        CUDA_CALL(cudaMallocManaged(&input_data, (long) input_n * input_c * input_h * input_w * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&d_input_data, (long) input_n * input_c * input_h * input_w * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&d_output_data, (long) input_n * input_c * input_h * input_w * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&bn_scale, (long) input_c * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&d_bn_scale, (long) input_c * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&d_bn_bias, (long) input_c * sizeof(float)));
        CPUFillRand(input_data, (long) input_n * input_c * input_h * input_w * sizeof(float));
        CPUFillRand(d_output_data, (long) input_n * input_c * input_h * input_w * sizeof(float));
        CPUFillRand(bn_scale, (long) input_c * sizeof(float));
        CPUFillRand(d_bn_scale, (long) input_c * sizeof(float));
        CPUFillRand(d_bn_bias, (long) input_c * sizeof(float));

        GPUFillRand(input_data, (long) input_n * input_c * input_h * input_w * sizeof(float) * input_ratio);
        GPUFillRand(d_output_data, (long) input_n * input_c * input_h * input_w * sizeof(float) * input_ratio);
        GPUFillRand(d_input_data, (long) input_n * input_c * input_h * input_w * sizeof(float) * output_ratio);
        GPUFillRand(bn_scale, (long) input_c * sizeof(float) * input_ratio);
        GPUFillRand(d_bn_scale, (long) input_c * sizeof(float) * output_ratio);
        GPUFillRand(d_bn_bias, (long) input_c * sizeof(float) * output_ratio);
        cudaDeviceSynchronize();
    }

    float one = 1;
    float zero = 0;
    
    float milliseconds = 0;
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));

    CUDA_CALL(cudaEventRecord(start));
    cudnnBatchNormalizationBackward(
        handle,
        CUDNN_BATCHNORM_SPATIAL,/* cudnnBatchNormMode_t mode */
        &one,
        &zero,
        &one,
        &zero,
        input_descriptor,       /* const cudnnTensorDescriptor_t xDesc */
        input_data,             /* const void *x */
        d_output_descriptor,      /* const cudnnTensorDescriptor_t yDesc */
        d_output_data,            /* void *y */
        d_input_descriptor,
        d_input_data,
        batch_norm_descriptor,  /* const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc */
        bn_scale,
        d_bn_scale,
        d_bn_bias,
        epsilon,
        nullptr,
        nullptr);
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));

    if (is_UVM) {
        CUDA_CALL(cudaFree(input_data));
        CUDA_CALL(cudaFree(d_input_data));
        CUDA_CALL(cudaFree(d_output_data));
        CUDA_CALL(cudaFree(d_bn_scale));
        CUDA_CALL(cudaFree(d_bn_bias));
    }
    return milliseconds;
}
