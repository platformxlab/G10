#pragma once
#include "Utils.cuh"
#include <vector>
#include <cudnn.h>

using std::vector;

class Concat_Backward : public Profiler {
    public:
        cudnnHandle_t handle;
        bool is_UVM;

        long input_n, output_c, input_h, input_w;
        long num_input;
        double input_ratio, output_ratio;
        long *input_cs;

        float **input_data;
        float **device_input_data;
        long *device_input_cs;
        float *output_data;

        Concat_Backward(cudnnHandle_t handle, vector<double> &args, bool is_UVM);
        ~Concat_Backward();
        float Run();
};
