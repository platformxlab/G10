#pragma once
#include "Utils.cuh"
#include <vector>
#include <cudnn.h>

using std::vector;

const static float dt = 1.0E-01f;
const static float threshold = 1.0E-02f;

class Add : public Profiler {
    public:
        cudnnHandle_t handle;
        bool is_UVM;

        long num_input, N;
        long batch_size, num_threads;
        double input_ratio, output_ratio;

        float **input_data;
        float **device_input_data;
        float *output_data;

        Add(cudnnHandle_t handle, vector<double> &args, bool is_UVM);
        ~Add();
        float Run();
};

class Add_Backward : public Profiler {
    public:
        cudnnHandle_t handle;
        bool is_UVM;

        long num_input, N;
        long batch_size, num_threads;
        double input_ratio, output_ratio;

        float **d_input_data;
        float **device_d_input_data;
        float *d_output_data;

        Add_Backward(cudnnHandle_t handle, vector<double> &args, bool is_UVM);
        ~Add_Backward();
        float Run();
};


class Divide_Forward : public Profiler {
    public:
        cudnnHandle_t handle;
        bool is_UVM;

        long N;
        long batch_size, num_threads;
        double input_ratio, output_ratio;

        float *input_a;
        float *input_b;
        float *output_c;

        Divide_Forward(cudnnHandle_t handle, vector<double> &args, bool is_UVM);
        ~Divide_Forward();
        float Run();
};



class Divide_Backward_A : public Profiler {
    public:
        cudnnHandle_t handle;
        bool is_UVM;

        long N;
        long batch_size, num_threads;
        double input_ratio, output_ratio;

        float *da;
        float *b;
        float *dc;

        Divide_Backward_A(cudnnHandle_t handle, vector<double> &args, bool is_UVM);
        ~Divide_Backward_A();
        float Run();
};



class Divide_Backward_B : public Profiler {
    public:
        cudnnHandle_t handle;
        bool is_UVM;

        long N;
        long batch_size, num_threads;
        double input_ratio, output_ratio;

        float *db;
        float *b;
        float *c;
        float *dc;

        Divide_Backward_B(cudnnHandle_t handle, vector<double> &args, bool is_UVM);
        ~Divide_Backward_B();
        float Run();
};




class Multiply : public Profiler {
    public:
        cudnnHandle_t handle;
        bool is_UVM;

        long N;
        long batch_size, num_threads;
        double input_ratio, output_ratio;

        float *output;
        float *inputA;
        float *inputB;

        Multiply(cudnnHandle_t handle, vector<double> &args, bool is_UVM);
        ~Multiply();
        float Run();
};



class Power_Forward : public Profiler {
    public:
        cudnnHandle_t handle;
        bool is_UVM;

        long N;
        long batch_size, num_threads;
        double input_ratio, output_ratio;

        float *output;
        float *inputA;
        float *inputB;

        Power_Forward(cudnnHandle_t handle, vector<double> &args, bool is_UVM);
        ~Power_Forward();
        float Run();
};




class Power_Backward : public Profiler {
    public:
        cudnnHandle_t handle;
        bool is_UVM;

        long N;
        long batch_size, num_threads;
        double input_ratio, output_ratio;

        float *da;
        float *inputA;
        float *inputB;
        float *dc;

        Power_Backward(cudnnHandle_t handle, vector<double> &args, bool is_UVM);
        ~Power_Backward();
        float Run();
};




class Sqrt_Forward : public Profiler {
    public:
        cudnnHandle_t handle;
        bool is_UVM;

        long N;
        long batch_size, num_threads;
        double input_ratio, output_ratio;

        float *output;
        float *input;

        Sqrt_Forward(cudnnHandle_t handle, vector<double> &args, bool is_UVM);
        ~Sqrt_Forward();
        float Run();
};


class Sqrt_Backward : public Profiler {
    public:
        cudnnHandle_t handle;
        bool is_UVM;

        long N;
        long batch_size, num_threads;
        double input_ratio, output_ratio;

        float *output_da;
        float *input_a;
        float *input_db;

        Sqrt_Backward(cudnnHandle_t handle, vector<double> &args, bool is_UVM);
        ~Sqrt_Backward();
        float Run();
};



class Tanh_Forward : public Profiler {
    public:
        cudnnHandle_t handle;
        bool is_UVM;

        long N;
        long batch_size, num_threads;
        double input_ratio, output_ratio;

        float *output;
        float *input;

        Tanh_Forward(cudnnHandle_t handle, vector<double> &args, bool is_UVM);
        ~Tanh_Forward();
        float Run();
};



class Tanh_Backward : public Profiler {
    public:
        cudnnHandle_t handle;
        bool is_UVM;

        long N;
        long batch_size, num_threads;
        double input_ratio, output_ratio;

        float *output_da;
        float *input_a;
        float *input_db;

        Tanh_Backward(cudnnHandle_t handle, vector<double> &args, bool is_UVM);
        ~Tanh_Backward();
        float Run();
};



class Erf_Forward : public Profiler {
    public:
        cudnnHandle_t handle;
        bool is_UVM;

        long N;
        long batch_size, num_threads;
        double input_ratio, output_ratio;

        float *output;
        float *input;

        Erf_Forward(cudnnHandle_t handle, vector<double> &args, bool is_UVM);
        ~Erf_Forward();
        float Run();
};



class Erf_Backward : public Profiler {
    public:
        cudnnHandle_t handle;
        bool is_UVM;

        long N;
        long batch_size, num_threads;
        double input_ratio, output_ratio;

        float *output_da;
        float *input_a;
        float *input_db;

        Erf_Backward(cudnnHandle_t handle, vector<double> &args, bool is_UVM);
        ~Erf_Backward();
        float Run();
};



class Sum_Forward : public Profiler {
    public:
        cudnnHandle_t handle;
        bool is_UVM;

        long N;
        long dim_3;
        long batch_size, num_threads;
        double input_ratio, output_ratio;

        float *output_b;
        float *input_a;

        Sum_Forward(cudnnHandle_t handle, vector<double> &args, bool is_UVM);
        ~Sum_Forward();
        float Run();
};


class Sum_Backward : public Profiler {
    public:
        cudnnHandle_t handle;
        bool is_UVM;

        long N;
        long dim_3;
        long batch_size, num_threads;
        double input_ratio, output_ratio;

        float *output_da;
        float *input_db;

        Sum_Backward(cudnnHandle_t handle, vector<double> &args, bool is_UVM);
        ~Sum_Backward();
        float Run();
};




class GatherV2_Forward : public Profiler {
    public:
        cudnnHandle_t handle;
        bool is_UVM;

        long N;
        long batch_size, num_threads;
        double input_ratio, output_ratio;

        float *output;
        float *input;

        GatherV2_Forward(cudnnHandle_t handle, vector<double> &args, bool is_UVM);
        ~GatherV2_Forward();
        float Run();
};



class GatherV2_Backward : public Profiler {
    public:
        cudnnHandle_t handle;
        bool is_UVM;

        long N;
        long batch_size, num_threads;
        double input_ratio, output_ratio;

        float *output_da;
        float *input_db;

        GatherV2_Backward(cudnnHandle_t handle, vector<double> &args, bool is_UVM);
        ~GatherV2_Backward();
        float Run();
};





class ApplyGrad : public Profiler {
    public:
        cudnnHandle_t handle;
        bool is_UVM;

        long N;
        long batch_size, num_threads;
        double input_ratio, output_ratio;

        float *output_data;
        float *d_output_data;

        ApplyGrad(cudnnHandle_t handle, vector<double> &args, bool is_UVM);
        ~ApplyGrad();
        float Run();
};

class MakeError : public Profiler {
    public:
        cudnnHandle_t handle;
        bool is_UVM;

        long N;
        long batch_size, num_threads;
        double input_ratio, output_ratio;

        float *error_data;
        float *output_data;

        MakeError(cudnnHandle_t handle, vector<double> &args, bool is_UVM);
        ~MakeError();
        float Run();
};

class TransferA0 : public Profiler {
    public:
        cudnnHandle_t handle;
        bool is_UVM;

        long N;
        long batch_size, num_threads;
        double input_ratio, output_ratio;

        float *data;
        float *activation_tensor;

        TransferA0(cudnnHandle_t handle, vector<double> &args, bool is_UVM);
        ~TransferA0();
        float Run();
};
