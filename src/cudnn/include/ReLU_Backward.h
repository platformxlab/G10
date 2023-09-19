#pragma once
#include "Utils.cuh"
#include <vector>
#include <cudnn.h>

using std::vector;

class ReLU_Backward : public Profiler {
    public:
        cudnnHandle_t handle;
        bool is_UVM;

        cudnnTensorDescriptor_t input_descriptor;
        cudnnTensorDescriptor_t d_input_descriptor;
        cudnnTensorDescriptor_t output_descriptor;
        cudnnTensorDescriptor_t d_output_descriptor;
        cudnnActivationDescriptor_t activation_descriptor;

        long input_n, input_c, input_h, input_w;
        double input_ratio, output_ratio;

        float* input_data;
        float* d_input_data;
        float* output_data;
        float* d_output_data;

        ReLU_Backward(cudnnHandle_t handle, vector<double> &args, bool is_UVM);
        ~ReLU_Backward();
        float Run();
};




class Softmax_Backward : public Profiler {
    public:
        cudnnHandle_t handle;
        bool is_UVM;

        cudnnTensorDescriptor_t input_descriptor;
        cudnnTensorDescriptor_t d_input_descriptor;

        cudnnTensorDescriptor_t d_output_descriptor;
        cudnnSoftmaxAlgorithm_t algorithm = CUDNN_SOFTMAX_FAST;
        cudnnSoftmaxMode_t      mode = CUDNN_SOFTMAX_MODE_INSTANCE;

        long input_n, input_c, input_h, input_w;
        double input_ratio, output_ratio;

        float* input_data;
        float* d_input_data;
        float* output_data;
        float* d_output_data;

        Softmax_Backward(cudnnHandle_t handle, vector<double> &args, bool is_UVM);
        ~Softmax_Backward();
        float Run();
};