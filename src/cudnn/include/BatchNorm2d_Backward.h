#pragma once
#include "Utils.cuh"
#include <vector>
#include <cudnn.h>

using std::vector;

class BatchNorm2d_Backward : public Profiler {
    public:
        cudnnHandle_t handle;
        bool is_UVM;

        cudnnTensorDescriptor_t input_descriptor;
        cudnnTensorDescriptor_t batch_norm_descriptor;
        cudnnTensorDescriptor_t d_input_descriptor;
        cudnnTensorDescriptor_t d_output_descriptor;

        long input_n, input_c, input_h, input_w;
        double input_ratio, output_ratio;

        float* bn_scale;
        float* d_bn_scale;
        float* d_bn_bias;
        const double exponentialAverageFactor = 0;
        const double epsilon = 0.0001;

        float* input_data;
        float* d_input_data;
        float* d_output_data;

        BatchNorm2d_Backward(cudnnHandle_t handle, vector<double> &args, bool is_UVM);
        ~BatchNorm2d_Backward();
        float Run();
};
