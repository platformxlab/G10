#pragma once
#include "Utils.cuh"
#include <vector>
#include <cudnn.h>

using std::vector;

class Linear_Backward_Bias : public Profiler {
    public:
        cudnnHandle_t handle;
        bool is_UVM;

        long input_n, input_c, input_h, input_w;
        long h_in, h_out;
        long reshape;

        float *d_bias;
        float *d_output;
        double input_ratio, output_ratio;

        Linear_Backward_Bias(cudnnHandle_t handle, vector<double> &args, bool is_UVM);
        ~Linear_Backward_Bias();
        float Run();
};
