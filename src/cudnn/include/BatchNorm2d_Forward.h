#pragma once
#include "Utils.cuh"
#include <vector>
#include <cudnn.h>

using std::vector;

class BatchNorm2d_Forward : public Profiler {
    public:
        cudnnHandle_t handle;
        bool is_UVM;

        cudnnTensorDescriptor_t input_descriptor;
        cudnnTensorDescriptor_t batch_norm_descriptor;
        cudnnTensorDescriptor_t output_descriptor;

        long input_n, input_c, input_h, input_w;
        double input_ratio, output_ratio;

        float* result_running_mean;
        float* result_running_variance;
        const double exponentialAverageFactor = 0;
        const double epsilon = 0.0001;

        float* input_data;
        float* output_data;
        float* bn_scale;
        float* bn_bias;

        BatchNorm2d_Forward(cudnnHandle_t handle, vector<double> &args, bool is_UVM);
        ~BatchNorm2d_Forward();
        float Run();
};
