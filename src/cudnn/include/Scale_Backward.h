#pragma once
#include "Utils.cuh"
#include <vector>
#include <cudnn.h>

using std::vector;

class Scale_Backward : public Profiler {
    public:
        cudnnHandle_t handle;
        bool is_UVM;

        long output_n, output_c, output_h, output_w;
        double input_ratio, output_ratio;

        float *d_input_data;
        float *d_output_data;

        Scale_Backward(cudnnHandle_t handle, vector<double> &args, bool is_UVM);
        ~Scale_Backward();
        float Run();
};
