#pragma once
#include "Utils.cuh"
#include <vector>
#include <cudnn.h>

using std::vector;

class Linear_Backward_Weight : public Profiler {
    public:
        cudnnHandle_t handle;
        bool is_UVM;

        long input_n, input_c, input_h, input_w;
        long h_in, h_out;
        long reshape;

        float *d_weight_data;
        float *input_data;
        float *d_output_data;
        double input_ratio, output_ratio;

        Linear_Backward_Weight(cudnnHandle_t handle, vector<double> &args, bool is_UVM);
        ~Linear_Backward_Weight();
        float Run();
};
