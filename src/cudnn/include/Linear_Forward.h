#pragma once
#include "Utils.cuh"
#include <vector>
#include <cudnn.h>

using std::vector;

class Linear_Forward : public Profiler {
    public:
        cudnnHandle_t handle;
        bool is_UVM;

        long input_n, input_c, input_h, input_w;
        long h_in, h_out;
        long reshape;

        float *input_data;
        float *weight_data;
        float *bias_data;
        float *output_data;
        double input_ratio, output_ratio;

        Linear_Forward(cudnnHandle_t handle, vector<double> &args, bool is_UVM);
        ~Linear_Forward();
        float Run();
};






class BatchMatMul_Forward : public Profiler {
    public:
        cudnnHandle_t handle;
        bool is_UVM;

        long input_n, input_c, input_h, input_w;
        long h_in, h_out;
        long reshape;

        float *input_data_A;
        float *input_data_B;

        float *output_data;
        double input_ratio, output_ratio;

        BatchMatMul_Forward(cudnnHandle_t handle, vector<double> &args, bool is_UVM);
        ~BatchMatMul_Forward();
        float Run();
};
