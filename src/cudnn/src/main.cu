#include "Conv2d_Forward.h"
#include "Conv2d_Backward_Weight.h"
#include "Conv2d_Backward_Input.h"
#include "BatchNorm2d_Forward.h"
#include "BatchNorm2d_Backward.h"
#include "Dropout_Forward.h"
#include "Dropout_Backward.h"
#include "MaxPool2d_Forward.h"
#include "MaxPool2d_Backward.h"
#include "AdaptiveAvgPool2d_Forward.h"
#include "AdaptiveAvgPool2d_Backward.h"
#include "ReLU_Forward.h"
#include "ReLU_Backward.h"
#include "Scale_Forward.h"
#include "Scale_Backward.h"
#include "Linear_Forward.h"
#include "Linear_Backward_Bias.h"
#include "Linear_Backward_Weight.h"
#include "Linear_Backward_Input.h"
#include "Concat_Forward.h"
#include "Concat_Backward.h"
#include "Others.h"
#include "Utils.cuh"
#include <assert.h>
#include <math.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <iterator>
#include <unistd.h>

// #define DEBUG_PRINT

int iterations;
int mean_iterations;

using std::stoi;
using std::stol;
using std::array;
using std::vector;
using std::string;
using std::ifstream;
using std::unique_ptr;
using std::runtime_error;

float get_avg(Profiler &p) {
    float total_runtime = 0;
#ifdef DEBUG_PRINT
    float min_runtime = 1e10, max_runtime = 0;
    vector<float> runtimes;
    float avg_diff = 0;
#endif
    for (int i = 0; i < iterations; i++) {
        float current_runtime = p.Run();
        if (i >= iterations - mean_iterations) {
            total_runtime += current_runtime;
#ifdef DEBUG_PRINT
            runtimes.push_back(current_runtime);
            min_runtime = current_runtime < min_runtime ? current_runtime : min_runtime;
            max_runtime = current_runtime > max_runtime ? current_runtime : max_runtime;
            printf("%10.4f  ", current_runtime);
#endif
        } else {
#ifdef DEBUG_PRINT
            printf("DISCARD: %10.4f\n", current_runtime);
#endif
        }
    }
#ifdef DEBUG_PRINT
    printf("\n");
#endif
    float avg_runtime = total_runtime / mean_iterations;
#ifdef DEBUG_PRINT
    float varience = 0;
    for (float runtime : runtimes) {
        printf("%10.4f  ", runtime / avg_runtime);
        varience += std::pow(runtime - avg_runtime, 2);
        avg_diff += std::abs(runtime - avg_runtime);
    }
    printf("\n");
    printf("Min: %f\n", min_runtime);
    printf("Max: %f\n", max_runtime);
    printf("Std: %f\n", std::sqrt(varience) / runtimes.size());
    printf("Avg: %f\n", avg_runtime);
    printf("Avg Diff: %f\n", avg_diff / runtimes.size());
    printf("Avg Diff%%: %f\n", avg_diff / runtimes.size() * 100);
#endif
    return avg_runtime;
}

float run_Conv2d_Forward(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM) {
    assert(args.size() == 13);
    Conv2d_Forward algo(cudnn, args, is_UVM);
    return get_avg(algo);
}

float run_ReLU_Forward(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM) {
    assert(args.size() == 6);
    ReLU_Forward algo(cudnn, args, is_UVM);
    return get_avg(algo);
}

float run_MaxPool2d_Forward(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM) {
    assert(args.size() == 12);
    MaxPool2d_Forward algo(cudnn, args, is_UVM);
    return get_avg(algo);
}

float run_AdaptiveAvgPool2d_Forward(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM) {
    assert(args.size() == 12);
    AdaptiveAvgPool2d_Forward algo(cudnn, args, is_UVM);
    return get_avg(algo);
}

float run_Linear_Forward(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM) {
    assert(args.size() == 8);
    Linear_Forward algo(cudnn, args, is_UVM);
    return get_avg(algo);
}

float run_Dropout_Forward(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM) {
    assert(args.size() == 6);
    Dropout_Forward algo(cudnn, args, is_UVM);
    return get_avg(algo);
}

float run_BatchNorm2d_Forward(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM) {
    assert(args.size() == 6);
    BatchNorm2d_Forward algo(cudnn, args, is_UVM);
    return get_avg(algo);
}

float run_Conv2d_Backward_Weight(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM) {
    assert(args.size() == 13);
    Conv2d_Backward_Weight algo(cudnn, args, is_UVM);
    return get_avg(algo);
}

float run_Conv2d_Backward_Input(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM) {
    assert(args.size() == 13);
    Conv2d_Backward_Input algo(cudnn, args, is_UVM);
    return get_avg(algo);
}

float run_Conv2d_Apply_Grad(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM) {
    assert(args.size() == 5);
    ApplyGrad algo(cudnn, args, is_UVM);
    return get_avg(algo);
}

float run_ReLU_Backward(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM) {
    assert(args.size() == 6);
    ReLU_Backward algo(cudnn, args, is_UVM);
    return get_avg(algo);
}

float run_MaxPool2d_Backward(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM) {
    assert(args.size() == 12);
    MaxPool2d_Backward algo(cudnn, args, is_UVM);
    return get_avg(algo);
}

float run_AdaptiveAvgPool2d_Backward(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM) {
    assert(args.size() == 12);
    AdaptiveAvgPool2d_Backward algo(cudnn, args, is_UVM);
    return get_avg(algo);
}

float run_Linear_Backward_Weight(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM) {
    assert(args.size() == 8);
    Linear_Backward_Weight algo(cudnn, args, is_UVM);
    return get_avg(algo);
}

float run_Linear_Backward_Input(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM) {
    assert(args.size() == 8);
    Linear_Backward_Input algo(cudnn, args, is_UVM);
    return get_avg(algo);
}

float run_Linear_Backward_Bias(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM) {
    assert(args.size() == 8);
    Linear_Backward_Bias algo(cudnn, args, is_UVM);
    return get_avg(algo);
}

float run_Linear_Apply_Grad_Bias(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM) {
    assert(args.size() == 5);
    ApplyGrad algo(cudnn, args, is_UVM);
    return get_avg(algo);
}

float run_Linear_Apply_Grad_Weight(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM) {
    assert(args.size() == 5);
    ApplyGrad algo(cudnn, args, is_UVM);
    return get_avg(algo);
}

float run_Dropout_Backward(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM) {
    assert(args.size() == 6);
    Dropout_Backward algo(cudnn, args, is_UVM);
    return get_avg(algo);
}

float run_BatchNorm2d_Backward(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM) {
    assert(args.size() == 6);
    BatchNorm2d_Backward algo(cudnn, args, is_UVM);
    return get_avg(algo);
}

float run_BatchNorm2d_Apply_Grad(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM) {
    assert(args.size() == 5);
    ApplyGrad algo(cudnn, args, is_UVM);
    return get_avg(algo);
}

float run_LoadData_A0(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM) {
    assert(args.size() == 3);
    TransferA0 algo(cudnn, args, is_UVM);
    return get_avg(algo);
}

float run_makeLoss(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM) {
    assert(args.size() == 5);
    MakeError algo(cudnn, args, is_UVM);
    return get_avg(algo);
}

float run_Add_Forward(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM) {
    assert(args.size() == 6);
    Add algo(cudnn, args, is_UVM);
    return get_avg(algo);
}

float run_Add_MultiGredient(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM) {
    assert(args.size() == 6);
    Add algo(cudnn, args, is_UVM);
    return get_avg(algo);
}

float run_Concat_Forward(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM) {
    assert(args.size() > 4 && args.size() == 6 + args[3]);
    Concat_Forward algo(cudnn, args, is_UVM);
    return get_avg(algo);
}

float run_Concat_Backward(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM) {
    assert(args.size() > 4 && args.size() == 6 + args[3]);
    Concat_Backward algo(cudnn, args, is_UVM);
    return get_avg(algo);
}

float run_Scale_Forward(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM) {
    assert(args.size() == 6);
    Scale_Forward algo(cudnn, args, is_UVM);
    return get_avg(algo);
}

float run_Scale_Backward(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM) {
    assert(args.size() == 6);
    Scale_Backward algo(cudnn, args, is_UVM);
    return get_avg(algo);
}


float run_GatherV2_Forward(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM){
    assert(args.size() == 5);
    GatherV2_Forward algo(cudnn, args, is_UVM);
    return get_avg(algo);
}


float run_GatherV2_Backward(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM){
    assert(args.size() == 5);
    GatherV2_Backward algo(cudnn, args, is_UVM);
    return get_avg(algo);
}



float run_Add_Backward(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM){
    assert(args.size() == 6);
    Add_Backward algo(cudnn, args, is_UVM);
    return get_avg(algo);
}


float run_Divide_Forward(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM){
    assert(args.size() == 5);
    Divide_Forward algo(cudnn, args, is_UVM);
    return get_avg(algo);
}


float run_Divide_Backward_A(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM){
    assert(args.size() == 5);
    Divide_Backward_A algo(cudnn, args, is_UVM);
    return get_avg(algo);
}


float run_Divide_Backward_B(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM){
    assert(args.size() == 5);
    Divide_Backward_B algo(cudnn, args, is_UVM);
    return get_avg(algo);
}


float run_Multiply_Forward(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM){
    assert(args.size() == 5);
    Multiply algo(cudnn, args, is_UVM);
    return get_avg(algo);
}


float run_Multiply_Backward(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM){
    assert(args.size() == 5);
    Divide_Forward algo(cudnn, args, is_UVM);
    return get_avg(algo);
}


float run_Power_Forward(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM){
    assert(args.size() == 5);
    Power_Forward algo(cudnn, args, is_UVM);
    return get_avg(algo);
}

float run_Power_Backward(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM){
    assert(args.size() == 5);
    Power_Backward algo(cudnn, args, is_UVM);
    return get_avg(algo);
}



float run_Sqrt_Forward(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM){
    assert(args.size() == 5);
    Sqrt_Forward algo(cudnn, args, is_UVM);
    return get_avg(algo);
}



float run_Sqrt_Backward(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM){
    assert(args.size() == 5);
    Sqrt_Backward algo(cudnn, args, is_UVM);
    return get_avg(algo);
}


float run_SoftmaxBasic_Forward(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM){
    assert(args.size() == 6);
    Softmax_Forward algo(cudnn, args, is_UVM);
    return get_avg(algo);
}



float run_SoftmaxBasic_Backward(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM){
    assert(args.size() == 6);
    Softmax_Backward algo(cudnn, args, is_UVM);
    return get_avg(algo);
}


float run_Sum_Forward(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM){
    assert(args.size() == 6);
    Sum_Forward algo(cudnn, args, is_UVM);
    return get_avg(algo);
}



float run_Sum_Backward(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM){
    assert(args.size() == 6);
    Sum_Backward algo(cudnn, args, is_UVM);
    return get_avg(algo);
}



float run_Tanh_Forward(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM){
    assert(args.size() == 5);
    Tanh_Forward algo(cudnn, args, is_UVM);
    return get_avg(algo);
}


float run_Tanh_Backward(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM){
    assert(args.size() == 5);
    Tanh_Backward algo(cudnn, args, is_UVM);
    return get_avg(algo);
}


float run_Erf_Forward(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM){
    assert(args.size() == 5);
    Erf_Forward algo(cudnn, args, is_UVM);
    return get_avg(algo);
}


float run_Erf_Backward(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM){
    assert(args.size() == 5);
    Erf_Backward algo(cudnn, args, is_UVM);
    return get_avg(algo);
}



float run_BatchMatMul_Forward(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM){
    assert(args.size() == 8);
    BatchMatMul_Forward algo(cudnn, args, is_UVM);
    return get_avg(algo);
}


float run_BatchMatMul_Backward(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM){
    assert(args.size() == 8);
    BatchMatMul_Forward algo(cudnn, args, is_UVM);
    return get_avg(algo);
}


float run_Subtract_Forward(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM){
    assert(args.size() == 6);
    Add algo(cudnn, args, is_UVM);
    return get_avg(algo);
}


float run_Subtract_Backward(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM){
    assert(args.size() == 6);
    Add_Backward algo(cudnn, args, is_UVM);
    return get_avg(algo);
}


float run_Apply_Grad(cudnnHandle_t &cudnn, vector<double> &args, bool is_UVM){
    assert(args.size() == 5);
    ApplyGrad algo(cudnn, args, is_UVM);
    return get_avg(algo);
}





float direct_run(cudnnHandle_t& cudnn, string algo, vector<double> &args, bool is_UVM) {
    float result;
    if (algo == "Conv2d_Forward")                   result = run_Conv2d_Forward(cudnn, args, is_UVM);
    else if (algo == "ReLU_Forward")                result = run_ReLU_Forward(cudnn, args, is_UVM);
    else if (algo == "MaxPool2d_Forward")           result = run_MaxPool2d_Forward(cudnn, args, is_UVM);
    else if (algo == "AdaptiveAvgPool2d_Forward")   result = run_AdaptiveAvgPool2d_Forward(cudnn, args, is_UVM);
    else if (algo == "Linear_Forward")              result = run_Linear_Forward(cudnn, args, is_UVM);
    else if (algo == "Dropout_Forward")             result = run_Dropout_Forward(cudnn, args, is_UVM);
    else if (algo == "BatchNorm2d_Forward")         result = run_BatchNorm2d_Forward(cudnn, args, is_UVM);
    else if (algo == "Conv2d_Backward_Weight")      result = run_Conv2d_Backward_Weight(cudnn, args, is_UVM);
    else if (algo == "Conv2d_Backward_Input")       result = run_Conv2d_Backward_Input(cudnn, args, is_UVM);
    else if (algo == "Conv2d_Apply_Grad")           result = run_Conv2d_Apply_Grad(cudnn, args, is_UVM);
    else if (algo == "ReLU_Backward")               result = run_ReLU_Backward(cudnn, args, is_UVM);
    else if (algo == "MaxPool2d_Backward")          result = run_MaxPool2d_Backward(cudnn, args, is_UVM);
    else if (algo == "AdaptiveAvgPool2d_Backward")  result = run_AdaptiveAvgPool2d_Backward(cudnn, args, is_UVM);
    else if (algo == "Linear_Backward_Weight")      result = run_Linear_Backward_Weight(cudnn, args, is_UVM);
    else if (algo == "Linear_Backward_Input")       result = run_Linear_Backward_Input(cudnn, args, is_UVM);
    else if (algo == "Linear_Backward_Bias")        result = run_Linear_Backward_Bias(cudnn, args, is_UVM);
    else if (algo == "Linear_Apply_Grad_Bias")      result = run_Linear_Apply_Grad_Bias(cudnn, args, is_UVM);
    else if (algo == "Linear_Apply_Grad_Weight")    result = run_Linear_Apply_Grad_Weight(cudnn, args, is_UVM);
    else if (algo == "Dropout_Backward")            result = run_Dropout_Backward(cudnn, args, is_UVM);
    else if (algo == "BatchNorm2d_Backward")        result = run_BatchNorm2d_Backward(cudnn, args, is_UVM);
    else if (algo == "BatchNorm2d_Apply_Grad")      result = run_BatchNorm2d_Apply_Grad(cudnn, args, is_UVM);
    else if (algo == "LoadData_A0")                 result = run_LoadData_A0(cudnn, args, is_UVM);
    else if (algo == "makeLoss")                    result = run_makeLoss(cudnn, args, is_UVM);
    else if (algo == "Add_Forward")                 result = run_Add_Forward(cudnn, args, is_UVM);
    else if (algo == "Add_MultiGredient")           result = run_Add_MultiGredient(cudnn, args, is_UVM);
    else if (algo == "Concat_Forward")              result = run_Concat_Forward(cudnn, args, is_UVM);
    else if (algo == "Concat_Backward")             result = run_Concat_Backward(cudnn, args, is_UVM);
    else if (algo == "Scale_Forward")               result = run_Scale_Forward(cudnn, args, is_UVM);
    else if (algo == "Scale_Backward")              result = run_Scale_Backward(cudnn, args, is_UVM);

    else if (algo == "GatherV2_Forward")              result = run_GatherV2_Forward(cudnn, args, is_UVM);
    else if (algo == "GatherV2_Backward")              result = run_GatherV2_Backward(cudnn, args, is_UVM);
    else if (algo == "Add_Backward")              result = run_Add_Backward(cudnn, args, is_UVM);
    else if (algo == "Divide_Forward")              result = run_Divide_Forward(cudnn, args, is_UVM);
    else if (algo == "Divide_Backward_A")              result = run_Divide_Backward_A(cudnn, args, is_UVM);
    else if (algo == "Divide_Backward_B")              result = run_Divide_Backward_B(cudnn, args, is_UVM);
    else if (algo == "Multiply_Forward")              result = run_Multiply_Forward(cudnn, args, is_UVM);
    else if (algo == "Multiply_Backward")              result = run_Multiply_Backward(cudnn, args, is_UVM);
    else if (algo == "Power_Forward")              result = run_Power_Forward(cudnn, args, is_UVM);
    else if (algo == "Power_Backward")              result = run_Power_Backward(cudnn, args, is_UVM);
    else if (algo == "Sqrt_Forward")              result = run_Sqrt_Forward(cudnn, args, is_UVM);
    else if (algo == "Sqrt_Backward")              result = run_Sqrt_Backward(cudnn, args, is_UVM);
    else if (algo == "SoftmaxBasic_Forward")              result = run_SoftmaxBasic_Forward(cudnn, args, is_UVM);
    else if (algo == "SoftmaxBasic_Backward")              result = run_SoftmaxBasic_Backward(cudnn, args, is_UVM);
    else if (algo == "Sum_Forward")              result = run_Sum_Forward(cudnn, args, is_UVM);
    else if (algo == "Sum_Backward")              result = run_Sum_Backward(cudnn, args, is_UVM);
    else if (algo == "Tanh_Forward")              result = run_Tanh_Forward(cudnn, args, is_UVM);
    else if (algo == "Tanh_Backward")              result = run_Tanh_Backward(cudnn, args, is_UVM);
    else if (algo == "Erf_Forward")              result = run_Erf_Forward(cudnn, args, is_UVM);
    else if (algo == "Erf_Backward")              result = run_Erf_Backward(cudnn, args, is_UVM);
    else if (algo == "BatchMatMul_Forward")              result = run_BatchMatMul_Forward(cudnn, args, is_UVM);
    else if (algo == "BatchMatMul_Backward")              result = run_BatchMatMul_Backward(cudnn, args, is_UVM);
    else if (algo == "Subtract_Forward")              result = run_Subtract_Forward(cudnn, args, is_UVM);
    else if (algo == "Subtract_Backward")              result = run_Subtract_Backward(cudnn, args, is_UVM);
    else if (algo == "Apply_Grad")              result = run_Apply_Grad(cudnn, args, is_UVM);

    else {
        assert(false);
    }
    return result;
}

void grouped_run(cudnnHandle_t& cudnn, string input_filename, bool is_UVM) {
    ifstream fin(input_filename);
    assert(fin.good());
    string line;
    string algo;
    int kernel_num = 0;

    string total_knum_str;
    array<char, 128> buffer;
    string cmd = "wc -l " + input_filename;
    unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) throw runtime_error("popen() failed!");
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        total_knum_str += buffer.data();
    }
    int total_size = std::ceil(std::log10(stoi(total_knum_str) - 1));

    while (std::getline(fin, line)) {
        std::stringstream ss(line);
        ss >> algo;
        vector<double> args((std::istream_iterator<long>(ss)), std::istream_iterator<long>());
        Profiler *p;
        if (algo == "Conv2d_Forward")                   p = new Conv2d_Forward(cudnn, args, is_UVM);
        else if (algo == "ReLU_Forward")                p = new ReLU_Forward(cudnn, args, is_UVM);
        else if (algo == "MaxPool2d_Forward")           p = new MaxPool2d_Forward(cudnn, args, is_UVM);
        else if (algo == "AdaptiveAvgPool2d_Forward")   p = new AdaptiveAvgPool2d_Forward(cudnn, args, is_UVM);
        else if (algo == "Linear_Forward")              p = new Linear_Forward(cudnn, args, is_UVM);
        else if (algo == "Dropout_Forward")             p = new Dropout_Forward(cudnn, args, is_UVM);
        else if (algo == "BatchNorm2d_Forward")         p = new BatchNorm2d_Forward(cudnn, args, is_UVM);
        else if (algo == "Conv2d_Backward_Weight")      p = new Conv2d_Backward_Weight(cudnn, args, is_UVM);
        else if (algo == "Conv2d_Backward_Input")       p = new Conv2d_Backward_Input(cudnn, args, is_UVM);
        else if (algo == "Conv2d_Apply_Grad")           p = new ApplyGrad(cudnn, args, is_UVM);
        else if (algo == "ReLU_Backward")               p = new ReLU_Backward(cudnn, args, is_UVM);
        else if (algo == "MaxPool2d_Backward")          p = new MaxPool2d_Backward(cudnn, args, is_UVM);
        else if (algo == "AdaptiveAvgPool2d_Backward")  p = new AdaptiveAvgPool2d_Backward(cudnn, args, is_UVM);
        else if (algo == "Linear_Backward_Weight")      p = new Linear_Backward_Weight(cudnn, args, is_UVM);
        else if (algo == "Linear_Backward_Input")       p = new Linear_Backward_Input(cudnn, args, is_UVM);
        else if (algo == "Linear_Backward_Bias")        p = new Linear_Backward_Bias(cudnn, args, is_UVM);
        else if (algo == "Linear_Apply_Grad_Bias")      p = new ApplyGrad(cudnn, args, is_UVM);
        else if (algo == "Linear_Apply_Grad_Weight")    p = new ApplyGrad(cudnn, args, is_UVM);
        else if (algo == "Dropout_Backward")            p = new Dropout_Backward(cudnn, args, is_UVM);
        else if (algo == "BatchNorm2d_Backward")        p = new BatchNorm2d_Backward(cudnn, args, is_UVM);
        else if (algo == "BatchNorm2d_Apply_Grad")      p = new ApplyGrad(cudnn, args, is_UVM);
        else if (algo == "LoadData_A0")                 p = new TransferA0(cudnn, args, is_UVM);
        else if (algo == "makeLoss")                    p = new MakeError(cudnn, args, is_UVM);
        else if (algo == "Add_Forward")                 p = new Add(cudnn, args, is_UVM);
        else if (algo == "Add_MultiGredient")           p = new Add(cudnn, args, is_UVM);
        else if (algo == "Concat_Forward")              p = new Concat_Forward(cudnn, args, is_UVM);
        else if (algo == "Concat_Backward")             p = new Concat_Backward(cudnn, args, is_UVM);
        else if (algo == "Scale_Forward")               p = new Scale_Forward(cudnn, args, is_UVM);
        else if (algo == "Scale_Backward")              p = new Scale_Backward(cudnn, args, is_UVM);
        else                                            assert(false);
        printf("%0*d %f ms\n", total_size, kernel_num++, get_avg(*p));
        delete p;
    }
}

size_t direct_get_workspace_size(cudnnHandle_t& cudnn, string algo, vector<double> &args) {
    size_t result;
    if (algo == "Conv2d_Forward")                   result = Conv2d_Forward(cudnn, args, 1).getWorkspaceSize();
    else if (algo == "ReLU_Forward")                result = ReLU_Forward(cudnn, args, 1).getWorkspaceSize();
    else if (algo == "MaxPool2d_Forward")           result = MaxPool2d_Forward(cudnn, args, 1).getWorkspaceSize();
    else if (algo == "AdaptiveAvgPool2d_Forward")   result = AdaptiveAvgPool2d_Forward(cudnn, args, 1).getWorkspaceSize();
    else if (algo == "Linear_Forward")              result = Linear_Forward(cudnn, args, 1).getWorkspaceSize();
    else if (algo == "Dropout_Forward")             result = Dropout_Forward(cudnn, args, 1).getWorkspaceSize();
    else if (algo == "BatchNorm2d_Forward")         result = BatchNorm2d_Forward(cudnn, args, 1).getWorkspaceSize();
    else if (algo == "Conv2d_Backward_Weight")      result = Conv2d_Backward_Weight(cudnn, args, 1).getWorkspaceSize();
    else if (algo == "Conv2d_Backward_Input")       result = Conv2d_Backward_Input(cudnn, args, 1).getWorkspaceSize();
    else if (algo == "Conv2d_Apply_Grad")           result = ApplyGrad(cudnn, args, 1).getWorkspaceSize();
    else if (algo == "ReLU_Backward")               result = ReLU_Backward(cudnn, args, 1).getWorkspaceSize();
    else if (algo == "MaxPool2d_Backward")          result = MaxPool2d_Backward(cudnn, args, 1).getWorkspaceSize();
    else if (algo == "AdaptiveAvgPool2d_Backward")  result = AdaptiveAvgPool2d_Backward(cudnn, args, 1).getWorkspaceSize();
    else if (algo == "Linear_Backward_Weight")      result = Linear_Backward_Weight(cudnn, args, 1).getWorkspaceSize();
    else if (algo == "Linear_Backward_Input")       result = Linear_Backward_Input(cudnn, args, 1).getWorkspaceSize();
    else if (algo == "Linear_Backward_Bias")        result = Linear_Backward_Bias(cudnn, args, 1).getWorkspaceSize();
    else if (algo == "Linear_Apply_Grad_Bias")      result = ApplyGrad(cudnn, args, 1).getWorkspaceSize();
    else if (algo == "Linear_Apply_Grad_Weight")    result = ApplyGrad(cudnn, args, 1).getWorkspaceSize();
    else if (algo == "Dropout_Backward")            result = Dropout_Backward(cudnn, args, 1).getWorkspaceSize();
    else if (algo == "BatchNorm2d_Backward")        result = BatchNorm2d_Backward(cudnn, args, 1).getWorkspaceSize();
    else if (algo == "BatchNorm2d_Apply_Grad")      result = ApplyGrad(cudnn, args, 1).getWorkspaceSize();
    else if (algo == "LoadData_A0")                 result = TransferA0(cudnn, args, 1).getWorkspaceSize();
    else if (algo == "makeLoss")                    result = MakeError(cudnn, args, 1).getWorkspaceSize();
    else if (algo == "Add_Forward")                 result = Add(cudnn, args, 1).getWorkspaceSize();
    else if (algo == "Add_MultiGredient")           result = Add(cudnn, args, 1).getWorkspaceSize();
    else if (algo == "Concat_Forward")              result = Concat_Forward(cudnn, args, 1).getWorkspaceSize();
    else if (algo == "Concat_Backward")             result = Concat_Backward(cudnn, args, 1).getWorkspaceSize();
    else if (algo == "Scale_Forward")               result = Scale_Forward(cudnn, args, 1).getWorkspaceSize();
    else if (algo == "Scale_Backward")              result = Scale_Backward(cudnn, args, 1).getWorkspaceSize();

    else if (algo == "GatherV2_Forward")            result = GatherV2_Forward(cudnn, args, 1).getWorkspaceSize();  
    else if (algo == "GatherV2_Backward")           result = GatherV2_Backward(cudnn, args, 1).getWorkspaceSize(); 
    else if (algo == "Add_Backward")                result = Add_Backward(cudnn, args, 1).getWorkspaceSize(); 
    else if (algo == "Divide_Forward")              result = Divide_Forward(cudnn, args, 1).getWorkspaceSize(); 
    else if (algo == "Divide_Backward_A")           result = Divide_Backward_A(cudnn, args, 1).getWorkspaceSize(); 
    else if (algo == "Divide_Backward_B")           result = Divide_Backward_B(cudnn, args, 1).getWorkspaceSize(); 
    else if (algo == "Multiply_Forward")            result = Multiply(cudnn, args, 1).getWorkspaceSize(); 
    else if (algo == "Multiply_Backward")           result = Divide_Forward(cudnn, args, 1).getWorkspaceSize(); 
    else if (algo == "Power_Forward")               result = Power_Forward(cudnn, args, 1).getWorkspaceSize(); 
    else if (algo == "Power_Backward")              result = Power_Backward(cudnn, args, 1).getWorkspaceSize(); 
    else if (algo == "Sqrt_Forward")                result = Sqrt_Forward(cudnn, args, 1).getWorkspaceSize(); 
    else if (algo == "Sqrt_Backward")               result = Sqrt_Backward(cudnn, args, 1).getWorkspaceSize(); 
    else if (algo == "SoftmaxBasic_Forward")        result = Softmax_Forward(cudnn, args, 1).getWorkspaceSize(); 
    else if (algo == "SoftmaxBasic_Backward")       result = Softmax_Backward(cudnn, args, 1).getWorkspaceSize(); 
    else if (algo == "Sum_Forward")                 result = Sum_Forward(cudnn, args, 1).getWorkspaceSize(); 
    else if (algo == "Sum_Backward")                result = Sum_Backward(cudnn, args, 1).getWorkspaceSize(); 
    else if (algo == "Tanh_Forward")                result = Tanh_Forward(cudnn, args, 1).getWorkspaceSize(); 
    else if (algo == "Tanh_Backward")               result = Tanh_Backward(cudnn, args, 1).getWorkspaceSize(); 
    else if (algo == "Erf_Forward")                 result = Erf_Forward(cudnn, args, 1).getWorkspaceSize(); 
    else if (algo == "Erf_Backward")                result = Erf_Backward(cudnn, args, 1).getWorkspaceSize(); 
    else if (algo == "BatchMatMul_Forward")         result = BatchMatMul_Forward(cudnn, args, 1).getWorkspaceSize(); 
    else if (algo == "BatchMatMul_Backward")        result = BatchMatMul_Forward(cudnn, args, 1).getWorkspaceSize(); 
    else if (algo == "Subtract_Forward")            result = Add(cudnn, args, 1).getWorkspaceSize(); 
    else if (algo == "Subtract_Backward")           result = Add_Backward(cudnn, args, 1).getWorkspaceSize(); 
    else if (algo == "Apply_Grad")                  result = ApplyGrad(cudnn, args, 1).getWorkspaceSize(); 
    else {
        assert(false);
    }
    return result;
}

void grouped_get_workspace_size(cudnnHandle_t& cudnn, string input_filename) {
    ifstream fin(input_filename);
    assert(fin.good());
    string line;
    string algo;
    int kernel_num = 0;

    string total_knum_str;
    array<char, 128> buffer;
    string cmd = "wc -l " + input_filename;
    unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) throw runtime_error("popen() failed!");
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        total_knum_str += buffer.data();
    }
    int total_size = std::ceil(std::log10(stoi(total_knum_str) - 1));

    while (std::getline(fin, line)) {
        std::stringstream ss(line);
        ss >> algo;
        vector<double> args((std::istream_iterator<long>(ss)), std::istream_iterator<long>());
        Profiler *p;
        if (algo == "Conv2d_Forward")                   p = new Conv2d_Forward(cudnn, args, 1);
        else if (algo == "ReLU_Forward")                p = new ReLU_Forward(cudnn, args, 1);
        else if (algo == "MaxPool2d_Forward")           p = new MaxPool2d_Forward(cudnn, args, 1);
        else if (algo == "AdaptiveAvgPool2d_Forward")   p = new AdaptiveAvgPool2d_Forward(cudnn, args, 1);
        else if (algo == "Linear_Forward")              p = new Linear_Forward(cudnn, args, 1);
        else if (algo == "Dropout_Forward")             p = new Dropout_Forward(cudnn, args, 1);
        else if (algo == "BatchNorm2d_Forward")         p = new BatchNorm2d_Forward(cudnn, args, 1);
        else if (algo == "Conv2d_Backward_Weight")      p = new Conv2d_Backward_Weight(cudnn, args, 1);
        else if (algo == "Conv2d_Backward_Input")       p = new Conv2d_Backward_Input(cudnn, args, 1);
        else if (algo == "Conv2d_Apply_Grad")           p = new ApplyGrad(cudnn, args, 1);
        else if (algo == "ReLU_Backward")               p = new ReLU_Backward(cudnn, args, 1);
        else if (algo == "MaxPool2d_Backward")          p = new MaxPool2d_Backward(cudnn, args, 1);
        else if (algo == "AdaptiveAvgPool2d_Backward")  p = new AdaptiveAvgPool2d_Backward(cudnn, args, 1);
        else if (algo == "Linear_Backward_Weight")      p = new Linear_Backward_Weight(cudnn, args, 1);
        else if (algo == "Linear_Backward_Input")       p = new Linear_Backward_Input(cudnn, args, 1);
        else if (algo == "Linear_Backward_Bias")        p = new Linear_Backward_Bias(cudnn, args, 1);
        else if (algo == "Linear_Apply_Grad_Bias")      p = new ApplyGrad(cudnn, args, 1);
        else if (algo == "Linear_Apply_Grad_Weight")    p = new ApplyGrad(cudnn, args, 1);
        else if (algo == "Dropout_Backward")            p = new Dropout_Backward(cudnn, args, 1);
        else if (algo == "BatchNorm2d_Backward")        p = new BatchNorm2d_Backward(cudnn, args, 1);
        else if (algo == "BatchNorm2d_Apply_Grad")      p = new ApplyGrad(cudnn, args, 1);
        else if (algo == "LoadData_A0")                 p = new TransferA0(cudnn, args, 1);
        else if (algo == "makeLoss")                    p = new MakeError(cudnn, args, 1);
        else if (algo == "Add_Forward")                 p = new Add(cudnn, args, 1);
        else if (algo == "Add_MultiGredient")           p = new Add(cudnn, args, 1);
        else if (algo == "Concat_Forward")              p = new Concat_Forward(cudnn, args, 1);
        else if (algo == "Concat_Backward")             p = new Concat_Backward(cudnn, args, 1);
        else if (algo == "Scale_Forward")               p = new Scale_Forward(cudnn, args, 1);
        else if (algo == "Scale_Backward")              p = new Scale_Backward(cudnn, args, 1);
        else                                            assert(false);
        printf("%0*d %lu B\n", total_size, kernel_num++, p->getWorkspaceSize());
        delete p;
    }
}

int main(int argc, char *argv[]) {
    assert(argc > 2);
    // argv[0]: program name    
    // argv[1]: kernel type/grouped run filename
    //          - if a kernel type is used, include its configurations in the argv[3...] field(s)
    //          - if a filename is used, do not include anything in the argv[3...] (i.e. call the file 
    //            with only 2 parameters)
    // argv[2]: other options
    //          0:  profiling without UVM, add "0 0" to the end of the kernel parameter list to indicate
    //              that both input and output are GPU resident at the start of kernel execution
    //          1:  profiling with UVM, specify the percentage of input/output that is not GPU resident
    //              at the end of the kernel parameter list
    //          2:  generate the workspace required for the kernel under specified parameters
    //          others: same as 1
    // argv[3...]: kernel parameters 
    string algo = string(argv[1]);
    bool is_UVM = stoi(string(argv[2])) != 0;
    bool workspace_generation = stoi(string(argv[2])) == 2;
    vector<double> args;
    for (int i = 3; i < argc; i++) args.push_back(stod(string(argv[i])));

    // hardcoded for now
    if (is_UVM) {
        iterations = 1;
        mean_iterations = 1;
    } else {
        iterations = 1;
        mean_iterations = 1;
    }
    assert(iterations >= mean_iterations);
    
    cudnnHandle_t cudnn;
    CUDNN_CALL(cudnnCreate(&cudnn));
    GPUInitRand();

    if (workspace_generation) {
        if (args.size() > 0) {
            size_t result = direct_get_workspace_size(cudnn, algo, args);
            printf("%lu B", result);
        } else {
            grouped_get_workspace_size(cudnn, algo);
        }
    } else {
        if (args.size() > 0) {
            float result = direct_run(cudnn, algo, args, is_UVM);
            printf("%f ms", result);
        } else {
            grouped_run(cudnn, algo, is_UVM);
        }
    }

    // Cleanup
    CUDNN_CALL(cudnnDestroy(cudnn));
}
