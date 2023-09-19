#pragma once
#include "Utils.cuh"
#include <vector>
#include <cudnn.h>

using std::vector;

class Dropout_Forward : public Profiler {
    public:
        cudnnHandle_t handle;
        bool is_UVM;

        cudnnTensorDescriptor_t input_descriptor;
        cudnnDropoutDescriptor_t dropout_descriptor;
        cudnnTensorDescriptor_t output_descriptor;

        size_t workspace_size;
        float *workspace_data;
        size_t state_size;
        float *state_data;

        long input_n, input_c, input_h, input_w;
        double input_ratio, output_ratio;

        const float dropout = 0.5f;
        const unsigned long long seed = 0;

        float *input_data;
        float *output_data;

        Dropout_Forward(cudnnHandle_t handle, vector<double> &args, bool is_UVM);
        ~Dropout_Forward();
        float Run();
        size_t getWorkspaceSize();
};
