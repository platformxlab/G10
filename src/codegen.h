#include <iostream>
#include <sstream>
#include <string>
#include <stdio.h>

using std::string;
using std::vector;

string conv2d_forward_CodeGen(int kernel_id, int input_height, int input_width,  int batch_size, int in_channels, int out_channels, int kernel_size_r, int kernel_size_s,
     int stride_0=1, int stride_1=1, int padding_0=0, int padding_1=0, bool bias=true);


string conv2d_backward_weight_CodeGen(int kernel_id, int input_height, int input_width, int batch_size, int in_channels, int out_channels, int kernel_size_r, int kernel_size_s,
     int stride_0=1, int stride_1=1, int padding_0=0, int padding_1=0, bool bias=true);


string conv2d_backward_input_CodeGen(int kernel_id, int input_height, int input_width, int batch_size, int in_channels, int out_channels, int kernel_size_r, int kernel_size_s,
    int stride_0=1, int stride_1=1, int padding_0=0, int padding_1=0, bool bias=true);


string maxPool2d_forward_CodeGen(int kernel_id, int batch_size, int in_channels, int input_height, int input_width, int kernel_size0, int kernel_size1, int stride, int padding, int dilation, bool ceil_mode);


string maxPool2d_backward_CodeGen(int kernel_id, int batch_size, int in_channels, int input_height, int input_width, int kernel_size0, int kernel_size1, int stride, int padding, int dilation, bool ceil_mode);


string reLU_forward_CodeGen(int kernel_id, int batch_size, int in_channels, int input_height, int input_width);


string reLU_backward_CodeGen(int kernel_id, int batch_size, int in_channels, int input_height, int input_width);


string avgAdaptivePool2d_forward_CodeGen(int kernel_id, int batch_size, int in_channels, int input_height, int input_width, int output_size_0, int output_size_1);


string avgAdaptivePool2d_backward_CodeGen(int kernel_id, int batch_size, int in_channels, int input_height, int input_width, int output_size_0, int output_size_1);


string linear_forward_CodeGen(int kernel_id, int batch_size, int in_channels, int input_height, int input_width, int H_in, int H_out);


string linear_backward_bias_CodeGen(int kernel_id, int batch_size, int in_channels, int input_height, int input_width, int H_in, int H_out);


string linear_backward_weight_CodeGen(int kernel_id, int batch_size, int in_channels, int input_height, int input_width, int H_in, int H_out);


string linear_backward_input_CodeGen(int kernel_id, int batch_size, int in_channels, int input_height, int input_width, int H_in, int H_out);


string batchnorm2d_forward_CodeGen(int kernel_id, int batch_size, int in_channels, int input_height, int input_width, float eps, float momentum, bool track_running);


string batchnorm2d_backward_CodeGen(int kernel_id, int batch_size, int in_channels, int input_height, int input_width);

string add_forward_CodeGen(int kernel_id, int num_inputs, int N);

string multiGradient_add_CodeGen(int kernel_id, int num_inputs, int N);

string concat_forward_CodeGen(int kernel_id, int batch_num, vector<int> &input_Cs, int input_height, int input_width);

string concat_backward_CodeGen(int kernel_id, int batch_num, vector<int> &input_Cs, int input_height, int input_width);

void main_code_generation();
void cudnn_profiling(bool individual_run = false, bool workspace_only = false);
