#include <assert.h>
#include <vector>
#include "Conv2d_Backward_Input.h"

Conv2d_Backward_Input::Conv2d_Backward_Input(cudnnHandle_t handle, vector<double> &args, bool is_UVM) : 
        handle(handle), is_UVM(is_UVM) {
    // 0. batch_size    1. in_channels   2. input_height  3. input_width
    // 4. out_channels  5. kernel_size_r 6. kernel_size_s
    // 7. padding0      8. padding1      9. stride0      10. stride1       
    input_n    = args[0];   input_c    = args[1];   input_h    = args[2];   input_w  = args[3];
    filter_n   = args[4];   filter_c   = args[1];   filter_h   = args[5];   filter_w = args[6];
    padding_h  = args[7];   padding_w  = args[8];
    stride_h   = args[9];   stride_w   = args[10];
    input_ratio = args[11]; output_ratio = args[12];
    dilation_h = 1;         dilation_w = 1;

    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
    CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_descriptor));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&d_output_descriptor));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&d_input_descriptor));
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&convolution_descriptor));

    // SetInputDescriptor
    CUDNN_CALL(cudnnSetTensor4dDescriptor(
            input_descriptor,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            input_n, input_c, input_h, input_w));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(
            d_input_descriptor,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            input_n, input_c, input_h, input_w));
    // SetFilterDescriptor
    CUDNN_CALL(cudnnSetFilter4dDescriptor(
            filter_descriptor,
            CUDNN_DATA_FLOAT,
            CUDNN_TENSOR_NCHW,
            filter_n, filter_c, filter_h, filter_w));
    // SetConvolutionDescriptor
    CUDNN_CALL(cudnnSetConvolution2dDescriptor(
            convolution_descriptor,
            padding_h, padding_w, stride_h, stride_w, dilation_h, dilation_w,
            CUDNN_CONVOLUTION,
            CUDNN_DATA_FLOAT));
    // SetOutputDescriptor
    CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(
            convolution_descriptor,
            input_descriptor, filter_descriptor,
            &output_n, &output_c, &output_h, &output_w));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(
            d_output_descriptor,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            output_n, output_c, output_h, output_w));
    // SetAlgorithm
    cudnnConvolutionBwdDataAlgoPerf_t convolution_algo_perf;
    int algo_count;
    cudnnGetConvolutionBackwardDataAlgorithm_v7(
            handle,
            filter_descriptor,
            d_output_descriptor,
            convolution_descriptor,
            d_input_descriptor,
            1,              /*requested algo count*/
            &algo_count,    /*returned algo count*/
            &convolution_algo_perf);

    algorithm = convolution_algo_perf.algo;
    CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
            handle,
            filter_descriptor, d_output_descriptor, convolution_descriptor, d_input_descriptor,
            algorithm,
            &workspace_size));
    // Alloc
    if (!is_UVM) {
        CUDA_CALL(cudaMalloc(&filter_data, (long) filter_n * filter_c * filter_h * filter_w * sizeof(float)));
        CUDA_CALL(cudaMalloc(&d_output_data, (long) output_n * output_c * output_h * output_w * sizeof(float)));
        CUDA_CALL(cudaMalloc(&d_input_data, (long) input_n * input_c * input_h * input_w * sizeof(float)));
        CUDA_CALL(cudaMalloc(&workspace_data, workspace_size));
        GPUFillRand(filter_data, (long) filter_n * filter_c * filter_h * filter_w * sizeof(float));
        GPUFillRand(d_output_data, (long) output_n * output_c * output_h * output_w * sizeof(float));
        GPUFillRand(workspace_data, workspace_size);
    }

    cudaDeviceSynchronize();
}

Conv2d_Backward_Input::~Conv2d_Backward_Input() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_descriptor));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(convolution_descriptor));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_descriptor));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(d_output_descriptor));
    if (!is_UVM) {
        CUDA_CALL(cudaFree(filter_data));
        CUDA_CALL(cudaFree(d_output_data));
        CUDA_CALL(cudaFree(d_input_data));
        CUDA_CALL(cudaFree(workspace_data));
    }
}

float Conv2d_Backward_Input::Run() {
    if (is_UVM) {
        CUDA_CALL(cudaMallocManaged(&filter_data, (long) filter_n * filter_c * filter_h * filter_w * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&d_output_data, (long) output_n * output_c * output_h * output_w * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&d_input_data, (long) input_n * input_c * input_h * input_w * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&workspace_data, workspace_size));
        CPUFillRand(filter_data, (long) filter_n * filter_c * filter_h * filter_w * sizeof(float));
        CPUFillRand(d_output_data, (long) output_n * output_c * output_h * output_w * sizeof(float));
        CPUFillRand(workspace_data, workspace_size);
        //std::cout<<"Workspace size:"<<workspace_size<<std::endl;
        GPUFillRand(d_output_data, (long) output_n * output_c * output_h * output_w * sizeof(float) * input_ratio);
        GPUFillRand(filter_data, (long) filter_n * filter_c * filter_h * filter_w * sizeof(float) * input_ratio);
        GPUFillRand(d_input_data, (long) input_n * input_c * input_h * input_w * sizeof(float) * output_ratio);
        GPUFillRand(workspace_data, workspace_size * output_ratio);
        cudaDeviceSynchronize();
    }

    float milliseconds = 0;
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));

    CUDA_CALL(cudaEventRecord(start));
    CUDNN_CALL(cudnnConvolutionBackwardData(
            handle,
            &alpha,
            filter_descriptor, filter_data,
            d_output_descriptor, d_output_data,
            convolution_descriptor, algorithm, workspace_data, workspace_size,
            &beta,
            d_input_descriptor, d_input_data));
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));

    if (is_UVM) {
        CUDA_CALL(cudaFree(filter_data));
        CUDA_CALL(cudaFree(d_output_data));
        CUDA_CALL(cudaFree(d_input_data));
        CUDA_CALL(cudaFree(workspace_data));
    }
    return milliseconds;
}

size_t Conv2d_Backward_Input::getWorkspaceSize() {
    return workspace_size;
}
