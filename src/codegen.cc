/* Note: We use cudnn and cublas now to generate the GPU program code for DNN training, but simple CUDA version is still available.*/

#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <regex>
#include <math.h>
#include <fstream>
#include <stdio.h>
#include "ast.h"
#include "analysis.h"
#include "printUtils.h"

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <array>

using std::map;
using std::ceil;
using std::array;
using std::regex;
using std::smatch;
using std::string;
using std::vector;
using std::ofstream;
using std::to_string;
using std::unique_ptr;
using std::runtime_error;

bool is_input_pf_only;

// see the meaning of the parameter argv2 in the file cudnn/main.cu main function documentation
static string exec(const CUDAKernelType type, const int argv2, const vector<long> &args) {
    array<char, 128> buffer;
    string result;
    string cmd = "./cudnn/main " + print_kerneltype_array[type] + " " + to_string(argv2);
    for (long arg : args)
        cmd += " " + to_string(arg);
    printf("%s: ", cmd.c_str());
    unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) throw runtime_error("popen() failed!");
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    printf("%s\n", result.c_str());
    return result;
}

static string exec(const string& filename, const int argv2) {
    array<char, 128> buffer;
    string result;
    string cmd = "stdbuf --output=L ./cudnn/main " + filename + " " + to_string(argv2);
    printf("%s:\n", cmd.c_str());
    unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) throw runtime_error("popen() failed!");
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        printf("%s", buffer.data());
        result += buffer.data();
    }
    return result;
}

string conv2d_forward_CodeGen(int kernel_id, int input_height, int input_width,  int batch_size, int in_channels, int out_channels, int kernel_size_r, int kernel_size_s,
     int stride_0=1, int stride_1=1, int padding_0=0, int padding_1=0, bool bias=true){

	//TODO: We don't care "bias" for now.
	char input_formal[50];
	char output_formal[50];
	char weight_formal[50];
	int output_P = ((input_height - kernel_size_r + 2 * padding_0)/stride_0) +1;
	int output_Q = ((input_width - kernel_size_s + 2 * padding_1)/stride_1) +1;
	sprintf(input_formal, "float input[%d][%d][%d][%d]", batch_size, in_channels, input_height, input_width);
	sprintf(output_formal, "float output[%d][%d][%d][%d]", batch_size, out_channels, output_P , output_Q);
	sprintf(weight_formal, "float weight[%d][%d][%d][%d]", out_channels, in_channels, kernel_size_r, kernel_size_s);
	string first_line = "__global__ void forwardPass_conv2d_l" + to_string(kernel_id) + "(" + string(input_formal) + ", " + string(output_formal) + ", " + string(weight_formal) + ")\n";
	string codegen = first_line;
	codegen += "{\n";
	codegen += "	const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;\n";
	codegen += "	const long parallel_size = blockDim.x * gridDim.x;\n";


	codegen += "	const long Z = " + to_string(batch_size) +"L*"+to_string(out_channels) +"*"+to_string(output_P)+"*"+to_string(output_Q)+";     //N * K * P*Q \n";
	codegen += "\n";
	codegen += "	for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {\n";
	codegen += "    long idx = n;\n";
	long arrayZ[5] = {1, batch_size, out_channels, output_P, output_Q};
	string arrayZ2[4] = {"n", "k", "p", "q"};
	for (long i = 1; i < 5; i++)
	{
		codegen += "    const long i" + to_string(i) +" = ((idx /= " + to_string(arrayZ[i-1]) + ") % " + to_string(arrayZ[i]) + ");     // " + arrayZ2[i-1]+"\n";
	}
	codegen += "    output[i1][i2][i3][i4] = 0;\n";
	codegen += "	}\n";


	codegen += "	__syncthreads();\n";


	codegen += "	const long N = " + to_string(batch_size) +"L*"+to_string(kernel_size_r)+"*"+to_string(kernel_size_s)+"*"+to_string(in_channels)+"*"+to_string(out_channels)
	+"*"+to_string(output_P)+"*"+to_string(output_Q)+";     //N * R*S * C * K * P*Q \n";
	codegen += "\n";
	codegen += "	for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {\n";
	codegen += "		long idx = n;\n";
	long array[8] = {1, batch_size, kernel_size_r, kernel_size_s, in_channels, out_channels, output_P, output_Q};
	string array2[7] = {"n", "r", "s", "c", "k", "p", "q"};
	for (long i = 1; i < 8; i++)
	{
		codegen += "		const long i" + to_string(i) +" = ((idx /= " + to_string(array[i-1]) + ") % " + to_string(array[i]) + ");     // " + array2[i-1]+"\n";
	}
	codegen += "		const long h = i6 * "+ to_string(stride_0)+" - "+ to_string(padding_0) +" + i2; // h = p * stride - pad + r;\n";
	codegen += "		const long w = i7 * "+ to_string(stride_1)+" - "+ to_string(padding_1) +" + i3; // w = q * stride - pad + s;\n";
	codegen += "\n";
	codegen += "		if (h >= 0 && h < " + to_string(input_height) + " && w >= 0 && w < " + to_string(input_width) + ")\n";
	codegen += "		{\n";
	codegen += " 			atomicAdd(&output[i1][i5][i6][i7], weight[i5][i4][i2][i3] * input[i1][i4][h][w]);\n";
	codegen += "		}\n";
	codegen += "	}\n";
	codegen += "}\n";

	return codegen;
}

/*      // Here is the generated example for (N, C, K, H, W, R, S) = (64, 1, 6, 28, 28, 7, 7), STRIDE=(1,1), PADDING=(2, 2), kernel_id=1:

__global__ void forwardPass_conv2d_l1(float input[64][1][28][28], float output[64][6][26][26], float weight[6][1][7][7])
{
	const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const long parallel_size = blockDim.x * gridDim.x;

	const long Z = 64*6*26*26;  //N * K * P*Q

	for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1	) % 64);  //n
		const long i2 = ((idx /= 64	) % 6);   //k
		const long i3 = ((idx /= 6	) % 26);   //p
		const long i4 = ((idx /= 26	) % 26);   //q

		output[i1][i2][i3][i4] = 0;
	}

	__syncthreads();


	const long N = 64*7*7*1*6*26*26;  //N * R*S * C * K * P*Q

	for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1	) % 64);  //n
		const long i2 = ((idx /= 64	) % 7);   //r
		const long i3 = ((idx /= 7	) % 7);   //s
		const long i4 = ((idx /= 7	) % 1);   //c
		const long i5 = ((idx /= 1	) % 6);   //k
		const long i6 = ((idx /= 6	) % 26);  //p
		const long i7 = ((idx /= 26	) % 26);  //q

		const long h = i6 * 1 - 2 + i2; // h = p * stride – pad + r;
		const long w = i7 * 1 - 2 + i3; // w = q * stride – pad + s;

		if (h >= 0 && h < 28 && w >= 0 && w < 28)
		{
			atomicAdd(&output[i1][i5][i6][i7], weight[i5][i4][i2][i3] * input[i1][i4][h][w]);
		}
	}
}
*/

string conv2d_backward_weight_CodeGen(int kernel_id, int input_height, int input_width,  int batch_size, int in_channels, int out_channels, int kernel_size_r, int kernel_size_s,
     int stride_0=1, int stride_1=1, int padding_0=0, int padding_1=0, bool bias=true){

  char input_formal[50];
	char output_formal[50];
	char weight_formal[50];

  int output_P = ((input_height - kernel_size_r + 2 * padding_0)/stride_0) +1;
  int output_Q = ((input_width - kernel_size_s + 2 * padding_1)/stride_1) +1;

  sprintf(input_formal, "float input[%d][%d][%d][%d]", batch_size, in_channels, input_height, input_width);
  sprintf(output_formal, "float d_output[%d][%d][%d][%d]", batch_size, out_channels, output_P, output_Q);
  sprintf(weight_formal, "float d_weight[%d][%d][%d][%d]", out_channels, in_channels, kernel_size_r, kernel_size_s);

  string first_line = "__global__ void backwardPass_conv2d_Weight_l" + to_string(kernel_id) + "(" + string(input_formal) + ", " + string(output_formal) + ", " + string(weight_formal) + ")\n";

  string codegen = first_line;
  codegen += "{\n";
  codegen += "  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;\n";
  codegen += "  const long parallel_size = blockDim.x * gridDim.x;\n";
  codegen += "\n";

  codegen += "  const long Z = " + to_string(out_channels) + "L*" + to_string(in_channels) + "*" + to_string(kernel_size_r) + "*" + to_string(kernel_size_s) + ";  // K * C * R * S\n";
  codegen += "\n";

  codegen += "  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {\n";
  codegen += "    long idx = n;\n";
  long arrayZ[5] = {1, out_channels, in_channels, kernel_size_r, kernel_size_s}; // k c r s
  string arrayZ2[4] = {"k", "c", "r", "s"};
  for(long i = 1; i < 5; ++i) {
    codegen += "    const long i" + to_string(i) + " = ((idx /= " + to_string(arrayZ[i-1]) + ") % " + to_string(arrayZ[i]) + ");   // " + arrayZ2[i-1] + "\n";
  }
  codegen += "\n";

  codegen += "    d_weight[i1][i2][i3][i4] = 0;\n";
  codegen += "  }\n";
  codegen += "\n";

  codegen += "  __syncthreads();\n";
  codegen += "\n";
  codegen += "\n";

  codegen += "  const long N = " + to_string(batch_size) + "L*" + to_string(kernel_size_r) + "*" + to_string(kernel_size_s) + "*" + to_string(in_channels) + "*" + to_string(out_channels) + "*" + to_string(output_P) + "*" + to_string(output_Q) + ";  // N * R*S * C * K * P*Q\n";
  codegen += "\n";

  codegen += "  for(long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {\n";
  codegen += "    long idx = n;\n";
  long array[8] = {1, batch_size, kernel_size_r, kernel_size_s, in_channels, out_channels, output_P, output_Q};
  string array2[7] = {"n", "r", "s", "c", "k", "p", "q"};
  for(long i = 1; i < 8; ++i) {
    codegen += "    const long i" + to_string(i) + " = ((idx /= " + to_string(array[i-1]) + "  ) % " + to_string(array[i]) + "); //" + array2[i-1] + "\n";
  }
  codegen += "\n";

  codegen += "    const long h = i6 * " + to_string(stride_0) + " - " + to_string(padding_0) + " + i2;  // h = p * stride - pad + r;\n";
  codegen += "    const long w = i7 * " + to_string(stride_1) + " - " + to_string(padding_1) + " + i3;  // w = q * stride - pad + s;\n";
  codegen += "\n";

  codegen += "    if(h >= 0 && h < " + to_string(input_height) + " && w >= 0 && w < " + to_string(input_width) + ")\n";
  codegen += "    {\n";
  codegen += "      atomicAdd(&d_weight[i5][i4][i2][i3], d_output[i1][i5][i6][i7] * input[i1][i4][h][w]);\n";
  codegen += "    }\n";
  codegen += "  }\n";
  codegen += "}\n";

  return codegen;
}
/*    // Here is the generated example for (N, C, K, H, W, R, S) = (64, 1, 6, 28, 28, 7, 7), STRIDE=(1,1), PADDING=(2, 2), kernel_id=1:

__global__ void backwardPass_conv2d_Weight_l1(float input[64][1][28][28], float d_output[64][6][26][26], float d_weight[6][1][7][7])
{
	const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const long parallel_size = blockDim.x * gridDim.x;

	const long Z = 6*1*7*7;  // K * C * R * S

	for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1	) % 6);  //k
		const long i2 = ((idx /= 6	) % 1);   //c
		const long i3 = ((idx /= 1	) % 7);   //r
		const long i4 = ((idx /= 7	) % 7);   //s

		d_weight[i1][i2][i3][i4] = 0;
	}

	__syncthreads();


	const long N = 64*7*7*1*6*26*26;  //N * R*S * C * K * P*Q

	for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1	) % 64);  //n
		const long i2 = ((idx /= 64	) % 7);   //r
		const long i3 = ((idx /= 7	) % 7);   //s
		const long i4 = ((idx /= 7	) % 1);   //c
		const long i5 = ((idx /= 1	) % 6);   //k
		const long i6 = ((idx /= 6	) % 26);  //p
		const long i7 = ((idx /= 26	) % 26);  //q

		const long h = i6 * 1 - 2 + i2; // h = p * stride – pad + r;
		const long w = i7 * 1 - 2 + i3; // w = q * stride – pad + s;

		if (h >= 0 && h < 28 && w >= 0 && w < 28)
		{
			atomicAdd(&d_weight[i5][i4][i2][i3], d_output[i1][i5][i6][i7] * input[i1][i4][h][w]);
		}
	}
}
*/

string conv2d_backward_input_CodeGen(int kernel_id, int input_height, int input_width,  int batch_size, int in_channels, int out_channels, int kernel_size_r, int kernel_size_s,
    int stride_0=1, int stride_1=1, int padding_0=0, int padding_1=0, bool bias=true){

  char input_formal[50];
  char output_formal[50];
  char weight_formal[50];

  int output_P = ((input_height - kernel_size_r + 2 * padding_0)/stride_0) +1;
  int output_Q = ((input_height - kernel_size_s + 2 * padding_1)/stride_1) +1;

  sprintf(input_formal, "float d_input[%d][%d][%d][%d]", batch_size, in_channels, input_height, input_width);
  sprintf(output_formal, "float d_output[%d][%d][%d][%d]", batch_size, out_channels, output_P, output_Q);
  sprintf(weight_formal, "float weight[%d][%d][%d][%d]", out_channels, in_channels, kernel_size_r, kernel_size_s);

  string first_line = "__global__ void backwardPass_conv2d_Input_l" + to_string(kernel_id) + "(" + input_formal + ", " + output_formal + ", " + weight_formal + ")\n";
  string codegen = first_line;

  codegen += "{\n";
  codegen += "  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;\n";
  codegen += "  const long parallel_size = blockDim.x * gridDim.x;\n";
  codegen += "\n";

  codegen += "  const long Z = " + to_string(batch_size) + "L*" + to_string(in_channels) + "*" + to_string(input_height) + "*" + to_string(input_width) + ";   // N * C * H * W\n";
  codegen += "\n";

  codegen += "  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {\n";
  codegen += "    long idx = n;\n";
  long arrayZ[5] = {1, batch_size, in_channels, input_height, input_width};
  string arrayZ2[4] = {"n", "c", "h", "w"};
  for(long i = 1; i < 5; ++i) {
    codegen += "    const long i" + to_string(i) + " = ((idx /= " + to_string(arrayZ[i-1]) + "   ) % " + to_string(arrayZ[i]) + ");  //" + arrayZ2[i-1] + "\n";
  }
  codegen += "\n";

  codegen += "    d_input[i1][i2][i3][i4] = 0;\n";
  codegen += "  }\n";
  codegen += "\n";

  codegen += "  __syncthreads();\n";
  codegen += "\n";

  codegen += "  const long N = " + to_string(batch_size) + "L*" + to_string(kernel_size_r) + "*" + to_string(kernel_size_s) + "*" + to_string(in_channels) + "*" + to_string(out_channels) + "*" + to_string(output_P) + "*" + to_string(output_Q) + ";   // N * R*S * C * K * P*Q\n";
  codegen += "\n";

  codegen += "  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {\n";
  codegen += "    long idx = n;\n";
  long array[8] = {1, batch_size, kernel_size_r, kernel_size_s, in_channels, out_channels, output_P, output_Q};
  string array2[7] = {"n", "r", "s", "c", "k", "p", "q"};
  for(long i = 1; i < 8; ++i) {
    codegen += "    const long i" + to_string(i) + " = ((idx /= " + to_string(array[i-1]) + "    ) % " + to_string(array[i]) + ");   //" + array2[i-1] + "\n";
  }
  codegen += "\n";

  codegen += "    const long h = i6 * " + to_string(stride_0) + " - " + to_string(padding_0) + " + i2; // h = p * stride - pad + r;\n";
  codegen += "    const long w = i7 * " + to_string(stride_1) + " - " + to_string(padding_1) + " + i3; // w = q * stride - pad + s;\n";
  codegen += "\n";

  codegen += "    if (h >= 0 && h < " + to_string(input_height) + " && w >= 0 && w < " + to_string(input_width) + ")\n";
  codegen += "    {\n";
  codegen += "      atomicAdd(&d_input[i1][i4][h][w], weight[i5][i4][i2][i3] * d_output[i1][i5][i6][i7]);\n";
  codegen += "    }\n";
  codegen += "  }\n";
  codegen += "}\n";

  return codegen;
}

/*    // Here is the generated example for (N, C, K, H, W, R, S) = (64, 1, 6, 28, 28, 7, 7), STRIDE=(1,1), PADDING=(2, 2), kernel_id=1:

__global__ void backwardPass_conv2d_Input_l1(float d_input[64][1][28][28], float d_output[64][6][26][26], float weight[6][1][7][7])
{
	const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const long parallel_size = blockDim.x * gridDim.x;

	const long Z = 64*1*28*28;  // N * C * H * W

	for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1	) % 64);  //n
		const long i2 = ((idx /= 64	) % 1);   //c
		const long i3 = ((idx /= 1	) % 28);   //h
		const long i4 = ((idx /= 28	) % 28);   //w

		d_input[i1][i2][i3][i4] = 0;
	}

	__syncthreads();


	const long N = 64*7*7*1*6*26*26;  //N * R*S * C * K * P*Q

	for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1	) % 64);  //n
		const long i2 = ((idx /= 64	) % 7);   //r
		const long i3 = ((idx /= 7	) % 7);   //s
		const long i4 = ((idx /= 7	) % 1);   //c
		const long i5 = ((idx /= 1	) % 6);   //k
		const long i6 = ((idx /= 6	) % 26);  //p
		const long i7 = ((idx /= 26	) % 26);  //q

		const long h = i6 * 1 - 2 + i2; // h = p * stride – pad + r;
		const long w = i7 * 1 - 2 + i3; // w = q * stride – pad + s;

		if (h >= 0 && h < 28 && w >= 0 && w < 28)
		{
			atomicAdd(&d_input[i1][i4][h][w], weight[i5][i4][i2][i3] * d_output[i1][i5][i6][i7]);
		}
	}
}

*/


string maxPool2d_forward_CodeGen(int kernel_id, int batch_size, int in_channels, int input_height, int input_width, int kernel_size0, int kernel_size1, int stride, int padding, int dilation, bool ceil_mode) {

  char input_formal[50];
  char output_formal[50];

  int output_P = ((input_height - kernel_size0 + 2 * padding)/stride) +1;
  int output_Q = ((input_width - kernel_size1 + 2 * padding)/stride) +1;

  sprintf(input_formal, "float input[%d][%d][%d][%d]", batch_size, in_channels, input_height, input_width);
  sprintf(output_formal, "float output[%d][%d][%d][%d]", batch_size, in_channels, output_P, output_Q);

  string first_line = "__global__ void forwardPass_MaxPool2d_l" + to_string(kernel_id) + "(" + input_formal + ", " + output_formal + ")\n";
  string codegen = first_line;
  codegen += "{\n";
  codegen += "  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;\n";
  codegen += "  const long parallel_size = blockDim.x * gridDim.x;\n";
  codegen += "\n";

  codegen += "  const long N = " + to_string(batch_size) + "L*" + to_string(in_channels) + "*" + to_string(output_P) + "*" + to_string(output_Q) + "; // N * K * P * Q\n";
  codegen += "\n";

  codegen += "  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {\n";
  codegen += "    long idx = n;\n";
  long array[5] = {1, batch_size, in_channels, output_P, output_Q};
  string array2[4] = {"n", "k", "p", "q"};
  for(long i = 1; i < 5; ++i) {
    codegen += "    const long i" + to_string(i) + " = ((idx /= " + to_string(array[i-1]) + "  ) % " + to_string(array[i]) + "); //" + array2[i-1] + "\n";
  }
  codegen += "\n";

  codegen += "    float max = -9999999;\n";
  codegen += "\n";

  codegen += "    for (long k0 = 0; k0 < " + to_string(kernel_size0) + "; k0++)  //kernel_size0\n";
  codegen += "    {\n";
  codegen += "      for (long k1 = 0; k1 < "+ to_string(kernel_size1) + "; k1++) //kernelsize_1\n";
  codegen += "      {\n";
  codegen += "        const long h = i3 * " + to_string(stride) + " - " + to_string(padding) + " + k0; // h = p * stride - pad + k0;\n";
  codegen += "        const long w = i4 * " + to_string(stride) + " - " + to_string(padding) + " + k1; // w = q * stride - pad + k1;\n";
  codegen += "        if (h >= 0 && h < " + to_string(input_height) + " && w >= 0 && w < " + to_string(input_width) + ")\n";
  codegen += "        {\n";
  codegen += "          max = max > input[i1][i2][h][w] ? max : input[i1][i2][h][w];\n";
  codegen += "        }\n";
  codegen += "      }\n";
  codegen += "    }\n";
  codegen += "    output[i1][i2][i3][i4] = max;\n";
  codegen += "  }\n";
  codegen += "}\n";

  return codegen;
}

/*
// kernel_id =2, (N,C,H,W) = (64, 6, 24, 24), kernel_size = (3,3), stride = 3, padding =0, dilation =1, ceil = false
__global__ void forwardPass_MaxPool2d_l2(float input[64][6][24][24], float output[64][6][8][8])
{
	const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const long parallel_size = blockDim.x * gridDim.x;


	const long N = 64*6*8*8;  // N * K * P * Q

	for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1	) % 64);  //n
		const long i2 = ((idx /= 64	) % 6);   //k
		const long i3 = ((idx /= 6	) % 8);   //p
		const long i4 = ((idx /= 8	) % 8);   //q

		float max = -9999999;

		for (long k0 = 0; k0 < 3; k0++)  //kernelsize_0
		{
			for (long k1 = 0; k1 < 3; k1++)  //kernelsize_1
			{
				const long h = i3 * 3 - 0 + k0; // h = p * stride – pad + k0;
				const long w = i4 * 3 - 0 + k1; // w = q * stride – pad + k1;
				if (h >= 0 && h < 24 && w >= 0 && w < 24)
				{
					max = max > input[i1][i2][h][w] ? max : input[i1][i2][h][w];
				}
			}
		}
		output[i1][i2][i3][i4] = max;
	} 

}
*/


string maxPool2d_backward_CodeGen(int kernel_id, int batch_size, int in_channels, int input_height, int input_width, int kernel_size0, int kernel_size1, int stride, int padding, int dilation, bool ceil_mode){
  char d_input_formal[50];
  char input_formal[50];
  char d_output_formal[50];

  int output_P = ((input_height - kernel_size0 + 2 * padding)/stride) +1;
  int output_Q = ((input_width - kernel_size1 + 2 * padding)/stride) +1;

  sprintf(d_input_formal, "float d_input[%d][%d][%d][%d]", batch_size, in_channels, input_height, input_width);
  sprintf(input_formal, "float input[%d][%d][%d][%d]", batch_size, in_channels, input_height, input_width);
  sprintf(d_output_formal, "float d_output[%d][%d][%d][%d]", batch_size, in_channels, output_P, output_Q);

  string first_line = "__global__ void backwardPass_MaxPool2d_l" + to_string(kernel_id) + "(" + d_input_formal + ", " + input_formal + ", " + d_output_formal + ")\n";
  string codegen = first_line;
  codegen += "{\n";
  codegen += "  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;\n";
  codegen += "  const long parallel_size = blockDim.x * gridDim.x;\n";
  codegen += "\n";

  codegen += "  const long Z = " + to_string(batch_size) + "L*" + to_string(in_channels) + "*" + to_string(input_height) + "*" + to_string(input_width) + ";  // N * C * H * W\n";
  codegen += "\n";

  codegen += "  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {\n";
  codegen += "    long idx = n;\n";
  long arrayZ[5] = {1, batch_size, in_channels, input_height, input_width};
  string arrayZ2[4] = {"n", "c", "h", "w"};
  for(long i = 1; i < 5; ++i) {
    codegen += "    const long i" + to_string(i) + " = ((idx /= " + to_string(arrayZ[i-1]) + "   ) % " + to_string(arrayZ[i]) + ");  //" + arrayZ2[i-1] + "\n";
  }
  codegen += "\n";

  codegen += "    d_input[i1][i2][i3][i4] = 0;\n";
  codegen += "  }\n";
  codegen += "\n";

  codegen += "  __syncthreads();\n";
  codegen += "\n";

  codegen += "  const long N = " + to_string(batch_size) + "L*" + to_string(in_channels) + "*" + to_string(output_P) + "*" + to_string(output_Q) + ";   // N * K * P * Q\n";
  codegen += "\n";

  codegen += "  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {\n";
  codegen += "    long idx = n;\n";
  long array[5] = {1, batch_size, in_channels, output_P, output_Q};
  string array2[4] = {"n", "c", "p", "q"};
  for(long i = 1; i < 5; ++i) {
    codegen += "    const long i" + to_string(i) + " = ((idx /= " + to_string(array[i-1]) + "  ) % " + to_string(array[i]) + ");   //" + array2[i-1] + "\n";
  }
  codegen += "\n";

  codegen += "    long max_idx_k0 = 0;\n";
  codegen += "    long max_idx_k1 = 0;\n";
  codegen += "    float max = -9999999;\n";
  codegen += "\n";

  codegen += "    for (long k0 = 0; k0 < " + to_string(kernel_size0) + "; k0++)  //kernelsize_0\n";
  codegen += "    {\n";
  codegen += "      for (long k1 = 0; k1 < " + to_string(kernel_size1) + "; k1++)  // kernelsize_1\n";
  codegen += "      {\n";
  codegen += "        const long h = i3 * " + to_string(stride) + " - " + to_string(padding) + " + k0; // h = p * stride - pad + k0;\n";
  codegen += "        const long w = i4 * " + to_string(stride) + " - " + to_string(padding) + " + k1; // w = q * stride - pad + k1;\n";
  codegen += "        if (h >= 0 && h < " + to_string(input_height) + " && w >= 0 && w < " + to_string(input_width) + ")\n";
  codegen += "        {\n";
  codegen += "          if (max = max < input[i1][i2][h][w])\n";
  codegen += "          {\n";
  codegen += "            max = input[i1][i2][h][w];\n";
  codegen += "            max_idx_k0 = k0;\n";
  codegen += "            max_idx_k1 = k1;      //Finding where the max value is\n";
  codegen += "          }\n";
  codegen += "        }\n";
  codegen += "      }\n";
  codegen += "    }\n";
  codegen += "\n";

  codegen += "    const long h_max = i3 * " + to_string(stride) + " - " + to_string(padding) + " + max_idx_k0;   // h = p * stride - pad + k0;\n";
  codegen += "    const long w_max = i4 * " + to_string(stride) + " - " + to_string(padding) + " + max_idx_k1;   // w = p * stride - pad + k1;\n";
  codegen += "\n";

  codegen += "    atomicAdd(&d_input[i1][i2][h_max][w_max], d_output[i1][i2][i3][i4]);\n";
  codegen += "  }\n";
  codegen += "}\n";

  return codegen;
}

/*
// (N,C,H,W) = (64, 6, 24, 24), kernel_size = (3,3), stride = 3, padding =0, dilation =1, ceil = false
__global__ void backwardPass_MaxPool2d_l2(float d_input[64][6][24][24], float input[64][6][24][24], float d_output[64][6][8][8])
{
	const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const long parallel_size = blockDim.x * gridDim.x;


	const long Z = 64*6*24*24;  // N * C * H * W

	for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1	) % 64);  //n
		const long i2 = ((idx /= 64	) % 6);   //c
		const long i3 = ((idx /= 6	) % 24);   //h
		const long i4 = ((idx /= 24	) % 24);   //w

		d_input[i1][i2][i3][i4] = 0;

	}

	__syncthreads();


	const long N = 64*6*8*8;  // N * K * P * Q

	for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1	) % 64);  //n
		const long i2 = ((idx /= 64	) % 6);   //c
		const long i3 = ((idx /= 6	) % 8);   //p
		const long i4 = ((idx /= 8	) % 8);   //q

		long max_idx_k0 = 0;
		long max_idx_k1 = 0;
		float max = -9999999;

		for (long k0 = 0; k0 < 3; k0++)  //kernelsize_0
		{
			for (long k1 = 0; k1 < 3; k1++)  //kernelsize_1
			{
				const long h = i3 * 3 - 0 + k0; // h = p * stride – pad + k0;
				const long w = i4 * 3 - 0 + k1; // w = q * stride – pad + k1;
				if (h >= 0 && h < 24 && w >= 0 && w < 24)
				{
					if (max = max < input[i1][i2][h][w])
					{
						max = input[i1][i2][h][w];
						max_idx_k0 = k0;
						max_idx_k1 = k1;         //Finding where the max value is
					}
				}
			}
		}

		const long h_max = i3 * 3 - 0 + max_idx_k0; // h = p * stride – pad + k0;
		const long w_max = i4 * 3 - 0 + max_idx_k1; // w = q * stride – pad + k1;

		atomicAdd(&d_input[i1][i2][h_max][w_max], d_output[i1][i2][i3][i4]);
	}

}
*/




string reLU_forward_CodeGen(int kernel_id, int batch_size, int in_channels, int input_height, int input_width){

  char input_formal[50];
  char output_formal[50];

  sprintf(input_formal, "float input[%d][%d][%d][%d]", batch_size, in_channels, input_height, input_width);
  sprintf(output_formal, "float output[%d][%d][%d][%d]", batch_size, in_channels, input_height, input_width);

  string first_line = "__global__ void forwardPass_ReLU_l" + to_string(kernel_id) + "(" + input_formal + ", " + output_formal + "){\n";
  string codegen = first_line;
  codegen += "\n";

  codegen += "  long size = " + to_string(batch_size) + "L*" + to_string(in_channels) + "*" + to_string(input_height) + "*" + to_string(input_width) + ";  //N * C * H * W;\n";
  codegen += "  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;\n";
  codegen += "  const long parallel_size = blockDim.x * gridDim.x;\n";
  codegen += "\n";

  codegen += "  const long Z = size;\n";
  codegen += "\n";

  codegen += "  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {\n";
  codegen += "    long idx = n;\n";
  long array[5] = {1, batch_size, in_channels, input_height, input_width};
  string array2[4] = {"n", "c", "h", "w"};
  for(long i = 1; i < 5; ++i) {
    codegen += "    const long i" + to_string(i) + " = ((idx /= " + to_string(array[i-1]) + "    ) % " + to_string(array[i]) + ");  //" + array2[i-1] + "\n";
  }
  codegen += "\n";

  codegen += "    output[i1][i2][i3][i4] = input[i1][i2][i3][i4] >= 0 ? input[i1][i2][i3][i4] : 0 ;\n";
  codegen += "  }\n";
  codegen += "}\n";

  return codegen;
}


/*  (N,C,H,W) = (64, 6, 24, 24), kernel_id =3;

__global__ void forwardPass_ReLU_l3(float input[64][6][24][24], float output[64][6][24][24]){

	long size = 64*6*24*24;  //N * C * H * W;
	const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const long parallel_size = blockDim.x * gridDim.x;


	const long Z = size;

	for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1	) % 64);  //n
		const long i2 = ((idx /= 64	) % 6);   //c
		const long i3 = ((idx /= 6	) % 24);   //h
		const long i4 = ((idx /= 24	) % 24);   //w

		output[i1][i2][i3][i4] = input[i1][i2][i3][i4] >= 0 ? input[i1][i2][i3][i4] : 0 ;
	}
}

*/


string reLU_backward_CodeGen(int kernel_id, int batch_size, int in_channels, int input_height, int input_width){

  char formal_input_d[50];
  char formal_input[50];
  char formal_output_d[50];

  sprintf(formal_input_d, "float d_input[%d][%d][%d][%d]", batch_size, in_channels, input_height, input_width);
  sprintf(formal_input, "float input[%d][%d][%d][%d]", batch_size, in_channels, input_height, input_width);
  sprintf(formal_output_d, "float d_output[%d][%d][%d][%d]", batch_size, in_channels, input_height, input_width);

  string first_line = "__global__ void backwardPass_ReLU_l" + to_string(kernel_id) + "(" + formal_input_d + ", " + formal_input + ", " + formal_output_d + ") {\n";
  string codegen = first_line;
  codegen += "\n";

  codegen += "  long size = " + to_string(batch_size) + "L*" + to_string(in_channels) + "*" + to_string(input_height) + "*" + to_string(input_width) + ";  //N * C * H * W;\n";
  codegen += "  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;\n";
  codegen += "  const long parallel_size = blockDim.x * gridDim.x;\n";
  codegen += "\n";

  codegen += "  const long Z = size;\n";
  codegen += "  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {\n";
  codegen += "    long idx = n;\n";
  long array[5] = {1, batch_size, in_channels, input_height, input_width};
  string array2[4] = {"n", "c", "h", "w"};
  for(long i = 1; i < 5; ++i) {
    codegen += "    const long i" + to_string(i) + " = ((idx /= " + to_string(array[i-1]) + "    ) % " + to_string(array[i]) + ");  //" + array2[i-1] + "\n";
  }
  codegen += "\n";

  codegen += "    d_input[i1][i2][i3][i4] = input[i1][i2][i3][i4] >= 0 ? d_output[i1][i2][i3][i4] : 0 ;\n";
  codegen += "  }\n";
  codegen += "}\n";

  return codegen;
}
/*  (N,C,H,W) = (64, 6, 24, 24), kernel_id = 3;

__global__ void backwardPass_ReLU_l3(float d_input[64][6][24][24], float input[64][6][24][24], float d_output[64][6][24][24]){

	long size = 64*6*24*24;  //N * C * H * W;
	const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const long parallel_size = blockDim.x * gridDim.x;


	const long Z = size;

	for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1	) % 64);  //n
		const long i2 = ((idx /= 64	) % 6);   //c
		const long i3 = ((idx /= 6	) % 24);   //h
		const long i4 = ((idx /= 24	) % 24);   //w

		d_input[i1][i2][i3][i4] = input[i1][i2][i3][i4] >= 0 ? d_output[i1][i2][i3][i4] : 0 ;
	}
}

*/



string avgAdaptivePool2d_forward_CodeGen(int kernel_id, int batch_size, int in_channels, int input_height, int input_width, int output_size_0, int output_size_1){

  char formal_input[50];
  char formal_output[50];

  sprintf(formal_input, "float input[%d][%d][%d][%d]", batch_size, in_channels, input_height, input_width);
  sprintf(formal_output, "float output[%d][%d][%d][%d]", batch_size, in_channels, output_size_0, output_size_1);

  string first_line = "__global__ void forwardPass_avgAdaptivePool2d_l" + to_string(kernel_id) + "(" + formal_input + ", " + formal_output + ") {\n";
  string codegen = first_line;
  codegen += "\n";

  codegen += "  long size = " + to_string(batch_size) + "L*" + to_string(in_channels) + "*" + to_string(output_size_0) + "*" + to_string(output_size_1) + ";  //N * C * S0 * S1;\n";
  codegen += "  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;\n";
  codegen += "  const long parallel_size = blockDim.x * gridDim.x;\n";
  codegen += "\n";

  codegen += "  const long Z = size;\n";
  codegen += "  const long ph = " + to_string(long(input_height / output_size_0)) + ";   // H/S0\n";
  codegen += "  const long pw = " + to_string(long(input_width / output_size_1)) + ";   // W/S1\n";
  codegen += "\n";

  codegen += "  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {\n";
  codegen += "    long idx = n;\n";
  long array[5] = {1, batch_size, in_channels, output_size_0, output_size_1};
  string array2[4] = {"n", "c", "p", "q"};
  for(long i = 1; i < 5; ++i) {
    codegen += "    const long i" + to_string(i) + " = ((idx /= " + to_string(array[i-1]) + "    ) % " + to_string(array[i]) + ");  //" + array2[i-1] + "\n";
  }
  codegen += "\n";

  codegen += "    float sum = 0;\n";
  codegen += "\n";

  codegen += "    for (long i = 0; i < ph; i++)\n";
  codegen += "    {\n";
  codegen += "      for (long j = 0; j < pw; j++)\n";
  codegen += "      {\n";
  codegen += "        sum += input[i1][i2][i3*ph+i][i4*pw+j];\n";
  codegen += "      }\n";
  codegen += "    }\n";
  codegen += "\n";

  codegen += "    output[i1][i2][i3][i4] = sum / (ph*pw);\n";
  codegen += "  }\n";
  codegen += "}\n";

  return codegen;
}


/*
(N,C,H,W) = (64, 6, 24, 24), kernel_id = 4, output_size = (8,8);

__global__ void forwardPass_avgAdaptivePool2d_l4(float input[64][6][24][24], float output[64][6][8][8]){

	long size = 64*6*8*8;  //N * C * S0 * S1;
	const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const long parallel_size = blockDim.x * gridDim.x;


	const long Z = size;
	const long ph = 3;   //  H/S0
	const long pw = 3;   //  H/S1

	for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1	) % 64);  //n
		const long i2 = ((idx /= 64	) % 6);   //c
		const long i3 = ((idx /= 6	) % 8);   //p
		const long i4 = ((idx /= 8	) % 8);   //q

		float sum = 0;

		for (long i = 0; i < ph; i++)
		{
			for (long j = 0; j < pw; j++)
			{
				sum += input[i1][i2][i3*ph+i][i4*pw+j];
			}

		}

		output[i1][i2][i3][i4] = sum / (ph*pw);

	}
}

*/


string avgAdaptivePool2d_backward_CodeGen(int kernel_id, int batch_size, int in_channels, int input_height, int input_width, int output_size_0, int output_size_1){

  char formal_input[50];
  char formal_output[50];

  sprintf(formal_input, "float d_input[%d][%d][%d][%d]", batch_size, in_channels, input_height, input_width);
  sprintf(formal_output, "float d_output[%d][%d][%d][%d]", batch_size, in_channels, output_size_0, output_size_1);

  string first_line = "__global__ void backwardPass_avgAdaptivePool2d_l" + to_string(kernel_id) + "(" + formal_input + ", " + formal_output + ") {\n";
  string codegen = first_line;
  codegen += "  long size = " + to_string(batch_size) + "L*" + to_string(in_channels) + "*" + to_string(input_height) + "*" + to_string(input_width) + ";  //N * C * H * W\n";
  codegen += "  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;\n";
  codegen += "  const long parallel_size = blockDim.x * gridDim.x;\n";
  codegen += "\n";

  codegen += "  const long Z = size;\n";
  codegen += "  const long ph = " + to_string(long(input_height / output_size_0)) + ";   // H/S0\n";
  codegen += "  const long pw = " + to_string(long(input_width / output_size_1)) + ";   // W/S1\n";
  codegen += "\n";

  codegen += "  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {\n";
  codegen += "    long idx = n;\n";
  long array[5] = {1, batch_size, in_channels, input_height, input_width};
  string array2[4] = {"n", "c", "h", "w"};
  for(long i = 1; i < 5; ++i) {
    codegen += "    const long i" + to_string(i) + " = ((idx /= " + to_string(array[i-1]) + "    ) % " + to_string(array[i]) + ");  //" + array2[i-1] + "\n";
  }
  codegen += "\n";

  codegen += "    d_input[i1][i2][i3][i4] = d_output[i1][i2][i3/ph][i4/ph] / (ph*pw);\n";
  codegen += "  }\n";
  codegen += "}\n";

  return codegen;
}

/*
__global__ void backwardPass_avgAdaptivePool2d_l4(float d_input[64][6][24][24], float d_output[64][6][8][8]){
  long size = 64*6*24*24;  //N * C * H * W;
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = size;
  const long ph = 3;
  const long pw = 3;

  for(long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1 ) % 64);  //n
    const long i2 = ((idx /= 64  ) % 6);   //c
    const long i3 = ((idx /= 6 ) % 24);   //h
    const long i4 = ((idx /= 24 ) % 24);   //w

    d_input[i1][i2][i3][i4] = d_output[i1][i2][i3/ph][i4/pw] / (ph*pw);
  }
}
*/


string linear_forward_CodeGen(int kernel_id, int batch_size, int in_channels, int input_height, int input_width, int H_in, int H_out){

  char formal_input[50];
  char formal_output[50];
  char formal_weight[50];
  char formal_bias[50];

  int reshape = batch_size*in_channels*input_height*input_width/H_in;

  sprintf(formal_input, "float input[%d][%d]", reshape , H_in);
  sprintf(formal_output, "float output[%d][%d]", reshape, H_out);
  sprintf(formal_weight, "float weight[%d][%d]", H_in, H_out);
  sprintf(formal_bias, "float bias[%d]", H_out);

  string first_line = "__global__ void forwardPass_Linear_l" + to_string(kernel_id) + "(" + formal_input + ", " + formal_output + ", " + formal_weight + ", " + formal_bias + ") {\n";
  string codegen = first_line;
  codegen += "  long size = " + to_string(reshape) + "L*" + to_string(H_out) + ";  //N*H_out\n";
  codegen += "  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;\n";
  codegen += "  const long parallel_size = blockDim.x * gridDim.x;\n";
  codegen += "\n";

  codegen += "  const long Z = size;\n";
  codegen += "\n";

  codegen += "  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {\n";
  codegen += "    long idx = n;\n";
  codegen += "    const long i1 = ((idx /= 1)  % " + to_string(reshape) + ");  //n\n";
  codegen += "    const long i2 = ((idx /= " + to_string(reshape) + ") % " + to_string(H_out) + ");  //h_out\n";
  codegen += "\n";

  codegen += "    output[i1][i2] = 0;\n";
  codegen += "    for (long k = 0; k < " + to_string(H_in) + "; ++k) {\n";
  codegen += "      output[i1][i2] += input[i1][k] * weight[k][i2];\n";
  codegen += "    }\n";
  codegen += "\n";
  codegen += "    output[i1][i2] += bias[i2];\n";
  codegen += "  }\n";
  codegen += "}\n";

  return codegen;
}

string linear_backward_bias_CodeGen(int kernel_id, int batch_size, int in_channels, int input_height, int input_width, int H_in, int H_out){

  char formal_output[50];
  char formal_bias[50];

  int reshape = batch_size*in_channels*input_height*input_width/H_in;

  sprintf(formal_output, "float d_output[%d][%d]", reshape, H_out);
  sprintf(formal_bias, "float d_bias[%d]", H_out);

  string first_line = "__global__ void backwardPass_Linear_bias_l" + to_string(kernel_id) + "(" + formal_output + ", " + formal_bias + ") {\n";
  string codegen = first_line;
  codegen += "  long size = " + to_string(H_out) + "L;  //H_out\n";
  codegen += "  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;\n";
  codegen += "  const long parallel_size = blockDim.x * gridDim.x;\n";
  codegen += "\n";

  codegen += "  const long Z = size;\n";
  codegen += "\n";

  codegen += "  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {\n";
  codegen += "    d_bias[n] = 0;\n";
  codegen += "\n";

  codegen += "    for (long i = 0; i < " + to_string(reshape) + "; ++i) {\n";
  codegen += "      d_bias[n] += d_output[i][n];\n";
  codegen += "    }\n";

  codegen += "  }\n";
  codegen += "}\n";

  return codegen;
}

string linear_backward_weight_CodeGen(int kernel_id, int batch_size, int in_channels, int input_height, int input_width, int H_in, int H_out){

  char formal_input[50];
  char formal_output[50];
  char formal_weight[50];


  int reshape = batch_size*in_channels*input_height*input_width/H_in;

  sprintf(formal_input, "float input[%d][%d]", reshape, H_in);
  sprintf(formal_output, "float d_output[%d][%d]", reshape, H_out);
  sprintf(formal_weight, "float d_weight[%d][%d]", H_in, H_out);


  string first_line = "__global__ void backwardPass_Linear_weight_l" + to_string(kernel_id) + "(" + formal_input + ", " + formal_output + ", " + formal_weight + ") {\n";
  string codegen = first_line;
  codegen += "\n";

  codegen += "  long size = " + to_string(H_in) + "L*" + to_string(H_out) + ";  //H_in * H_out\n";
  codegen += "  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;\n";
  codegen += "  const long parallel_size = blockDim.x * gridDim.x;\n";
  codegen += "\n";

  codegen += "  const long Z = size;\n";
  codegen += "\n";

  codegen += "  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {\n";
  codegen += "    long idx = n;\n";
  codegen += "    const long i1 = ((idx /= 1   ) % " + to_string(H_in) + ");  //H_in\n";
  codegen += "    const long i2 = ((idx /= " + to_string(H_in) + "   ) % " + to_string(H_out) + ");  //H_out\n";
  codegen += "    d_weight[i1][i2] = 0;\n";
  codegen += "\n";

  codegen += "    for (long k = 0; k < " + to_string(reshape) + "; ++k) {\n";
  codegen += "      d_weight[i1][i2] += input[k][i1] * d_output[k][i2];\n";
  codegen += "    }\n";
  codegen += "  }\n";
  codegen += "}\n";

  return codegen;
}

/* __global__ void backwardPass_Linear_weight_l5(d_input[64][128], d_output[64][200], d_weight[128][200], d_bias[200]) {

  long size = 128*200;  //H_in * H_out
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = size;

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1   ) % 128);  //H_in
    const long i2 = ((idx /= 128   ) % 200);  //H_out
    d_weight[i1][i2] = 0;

    for (long k = 0; k < 64; ++k) {
      d_weight[i1][i2] += d_input[k][i1] * d_output[k][i2];
    }
  }
}
*/

string linear_backward_input_CodeGen(int kernel_id, int batch_size, int in_channels, int input_height, int input_width, int H_in, int H_out) {

  char formal_input[50];
  char formal_output[50];
  char formal_weight[50];

  int reshape = batch_size*in_channels*input_height*input_width/H_in;

  sprintf(formal_input, "float d_input[%d][%d]", reshape, H_in);
  sprintf(formal_output, "float d_output[%d][%d]", reshape, H_out);
  sprintf(formal_weight, "float weight[%d][%d]", H_in, H_out);


  string first_line = "__global__ void backwardPass_Linear_input_l" + to_string(kernel_id) + "(" + formal_input + ", " + formal_output + ", " + formal_weight + ") {\n";
  string codegen = first_line;

  codegen += "  long size = " + to_string(reshape) + "L*" + to_string(H_in) + ";  //N * H_in\n";
  codegen += "  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;\n";
  codegen += "  const long parallel_size = blockDim.x * gridDim.x;\n";
  codegen += "\n";

  codegen += "  const long Z = size;\n";
  codegen += "\n";

  codegen += "  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {\n";
  codegen += "    long idx = n;\n";
  codegen += "    const long i1 = ((idx /= 1) % " + to_string(reshape) + ");  //n\n";
  codegen += "    const long i2 = ((idx /= " + to_string(reshape) + ") % " + to_string(H_in) + "); //h_in\n"; // TODO
  codegen += "\n";

  codegen += "    d_input[i1][i2] = 0;\n";
  codegen += "    for (long k = 0; k < " + to_string(H_out) + "; ++k) {\n";
  codegen += "      d_input[i1][i2] += d_output[i1][k] * weight[i2][k];\n";
  codegen += "    }\n";
  codegen += "  }\n";
  codegen += "}\n";

  return codegen;
}


/* __global__ void backwardPass_Linear_input_l5(d_input[64][128], d_output[64][200], d_weight[128][200], d_bias[200]) {
  long size = 64*128;  //N * H_in
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = size;

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64);  //n
    const long i2 = ((idx /= 64) % 128); //h_in

    d_input[i1][i2] = 0;
    for (long k = 0; k < 200; ++k) {
      d_input[i1][i2] += d_output[i1][k] * d_weight[i2][k];
    }
  }
}
*/




string batchnorm2d_forward_CodeGen(int kernel_id, int batch_size, int in_channels, int input_height, int input_width, float eps, float momentum, bool track_running) {
  char formal_input[50];
  char formal_gamma_and_beta[50];
  char formal_running_m[50];
  char formal_running_v[50];
  char formal_mu[50];
  char formal_var[50];
  char formal_v1[50];
  char formal_v2[50];
  char formal_output[50];

  sprintf(formal_input, "float input[%d][%d][%d][%d]", batch_size, in_channels, input_height, input_width);
  sprintf(formal_gamma_and_beta, "float gamma_and_beta[%d]", in_channels*2);
  sprintf(formal_running_m, "float running_m[%d]", in_channels);
  sprintf(formal_running_v, "float running_v[%d]", in_channels);
  sprintf(formal_mu, "float mu[%d]", in_channels);
  sprintf(formal_var, "float var[%d]", in_channels);
  sprintf(formal_v1, "float v1[%d][%d][%d][%d]", batch_size, in_channels, input_height, input_width);
  sprintf(formal_v2, "float v2[%d]", in_channels);
  sprintf(formal_output, "float output[%d][%d][%d][%d]", batch_size, in_channels, input_height, input_width);

  string first_line = "__global__ void forwardPass_BatchNorm2d_l" + to_string(kernel_id) + "(" + formal_input + ", " + formal_gamma_and_beta + ", " + formal_running_m + ", " + formal_running_v + ", " + formal_mu + ", " + formal_var + ", " + formal_v1 + ", " + formal_v2 + ", " + formal_output + ") { \n";
  string codegen = first_line;

  //TODO:
  codegen += "  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;\n";
  codegen += "  const long parallel_size = blockDim.x * gridDim.x;\n";
  codegen += "\n";
  codegen += "  const long Z1 = " + to_string(in_channels) + "L;  //C\n";
  codegen += "\n";
  codegen += "  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {\n";
  codegen += "    long idx = n;\n";
  codegen += "    const long i1 = ((idx /= 1) % " + to_string(in_channels) + ");  // c\n";
  codegen += "\n";
  codegen += "    mu[i1] = 0;\n";
  codegen += "    var[i1] = 0;\n";
  codegen += "  }\n";
  codegen += "\n";
  codegen += "  __syncthreads();\n";
  codegen += "\n";
  codegen += "  const long N1 = " + to_string(batch_size) + "L*" + to_string(in_channels) + "*" + to_string(input_height) + "*" + to_string(input_width) + ";  // N * C * H * W\n";
  codegen += "\n";
  codegen += "  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {\n";
  codegen += "    long idx = n;\n";
  codegen += "    const long i1 = ((idx /= 1) % " + to_string(batch_size) + "); // n\n";
  codegen += "    const long i2 = ((idx /= " + to_string(batch_size) + ") % " + to_string(in_channels) + "); // c\n";
  codegen += "    const long i3 = ((idx /= " + to_string(in_channels) + ") % " + to_string(input_height) + "); // h\n";
  codegen += "    const long i4 = ((idx /= " + to_string(input_height) + ") % " + to_string(input_width) + "); // w\n";
  codegen += "\n";
  codegen += "    atomicAdd(&mu[i2], input[i1][i2][i3][i4]);\n";
  codegen += "  }\n";
  codegen += "\n";
  codegen += "  __syncthreads();\n";
  codegen += "\n";
  codegen += "\n";
  codegen += "  const long N_bn = " + to_string(batch_size) + "L*" + to_string(input_height) + "*" + to_string(input_width) + ";  // N * H * W\n";
  codegen += "  const long N2 = " + to_string(in_channels) + "; // C\n";
  codegen += "  \n";
  codegen += "  for (long n = 0; n < N2; n += parallel_size) {\n";
  codegen += "    long idx = n + thread_pos;\n";
  codegen += "    if (idx < N2) {\n";
  codegen += "      mu[idx] = mu[idx] / N_bn;\n";
  codegen += "    }\n";
  codegen += "  }\n";
  codegen += "\n";
  codegen += "  __syncthreads();\n";
  codegen += "\n";
  codegen += "  for (long n = 0; n < N2; n += parallel_size) {\n";
  codegen += "    long idx = n + thread_pos;\n";
  codegen += "    if (idx < N2) {\n";
  codegen += "      running_m[idx] = " + to_string(momentum) + " * running_m[idx] + " + to_string(1 - momentum) + " * mu[idx];    // running_m = momentum * running_m + (1-momentum) * mu;\n";
  codegen += "    }\n";
  codegen += "  }\n";
  codegen += "\n";
  codegen += "  __syncthreads();\n";
  codegen += "\n";
  codegen += "  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {\n";
  codegen += "    long idx = n;\n";
  codegen += "    const long i1 = ((idx /= 1) % " + to_string(batch_size) + "); // n\n";
  codegen += "    const long i2 = ((idx /= " + to_string(batch_size) + ") % " + to_string(in_channels) + "); // c\n";
  codegen += "    const long i3 = ((idx /= " + to_string(in_channels) + ") % " + to_string(input_height) + "); // h\n";
  codegen += "    const long i4 = ((idx /= " + to_string(input_height) + ") % " + to_string(input_width) + "); // w\n";
  codegen += "\n";
  codegen += "    v1[i1][i2][i3][i4] = input[i1][i2][i3][i4] - running_m[i2];\n";
  codegen += "\n";
  codegen += "  }\n";
  codegen += "\n";
  codegen += "  __syncthreads();\n";
  codegen += "\n";
  codegen += "  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {\n";
  codegen += "    long idx = n;\n";
  codegen += "    const long i1 = ((idx /= 1) % " + to_string(batch_size) + "); // n\n";
  codegen += "    const long i2 = ((idx /= " + to_string(batch_size) + ") % " + to_string(in_channels) + "); // c\n";
  codegen += "    const long i3 = ((idx /= " + to_string(in_channels) + ") % " + to_string(input_height) + "); // h\n";
  codegen += "    const long i4 = ((idx /= " + to_string(input_height) + ") % " + to_string(input_width) + "); // w\n";
  codegen += "\n";
  codegen += "    atomicAdd(&var[i2], (v1[i1][i2][i3][i4]*v1[i1][i2][i3][i4]));\n";
  codegen += "\n";
  codegen += "  }\n";
  codegen += "\n";
  codegen += "  __syncthreads();\n";
  codegen += "\n";
  codegen += "\n";
  codegen += "  for (long n = 0; n < N2; n += parallel_size) {\n";
  codegen += "    long idx = n + thread_pos;\n";
  codegen += "    if (idx < N2) {\n";
  codegen += "      var[idx] = var[idx] / N_bn;\n";
  codegen += "    }\n";
  codegen += "  }\n";
  codegen += "\n";
  codegen += "  __syncthreads();\n";
  codegen += "\n";
  codegen += "  for (long n = 0; n < N2; n += parallel_size) {\n";
  codegen += "    long idx = n + thread_pos;\n";
  codegen += "    if (idx < N2) {\n";
  codegen += "      running_v[idx] = " + to_string(momentum) + " * running_v[idx] + "
  + to_string(1 - momentum) +" * var[idx];   // running_v = momentum * running_v + (1-momentum) * var;\n";
  codegen += "    }\n";
  codegen += "  }\n";
  codegen += "\n";
  codegen += "  __syncthreads();\n";
  codegen += "\n";
  codegen += "  for (long n = N2 * thread_pos / parallel_size; n < N2 * (thread_pos+1) / parallel_size; ++n) {\n";
  codegen += "    long idx = n;\n";
  codegen += "    const long i1 = ((idx /= 1) % " + to_string(in_channels) + ");  // c\n";
  codegen += "\n";
  codegen += "    v2[i1] = sqrt(running_v[i1] + " + to_string(eps) + "); //eps = 0.00001, v2 = sqrt(running_v + eps)\n";
  codegen += "  }\n";
  codegen += "\n";
  codegen += "  __syncthreads();\n";
  codegen += "\n";
  codegen += "  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {\n";
  codegen += "    long idx = n;\n";
  codegen += "    const long i1 = ((idx /= 1) % " + to_string(batch_size) + "); // n\n";
  codegen += "    const long i2 = ((idx /= " + to_string(batch_size) + ") % " + to_string(in_channels) + "); // c\n";
  codegen += "    const long i3 = ((idx /= " + to_string(in_channels) + ") % " + to_string(input_height) + "); // h\n";
  codegen += "    const long i4 = ((idx /= " + to_string(input_height) + ") % " + to_string(input_width) + "); // w\n";
  codegen += "\n";
  codegen += "    output[i1][i2][i3][i4] = (v1[i1][i2][i3][i4] / v2[i2]) * gamma_and_beta[i2] + gamma_and_beta[" + to_string(in_channels) + " + i2]; //gamma = gamma_and_beta[0~c-1], beta = gamma_and_beta[c~2c-1]; output = (v1/v2) * gamma + beta\n";
  codegen += "  }\n";

  codegen += "}\n";
  return codegen;
}

string batchnorm2d_backward_CodeGen(int kernel_id, int batch_size, int in_channels, int input_height, int input_width) {
  char d_output[50];
  char v1[50];
  char v2[50];
  char gamma_and_beta[50];
  char dv1[50];
  char dv2[50];
  char dvar[50];
  char dmu[50];
  char d_input[50];
  char d_gamma_and_beta[50];

  sprintf(d_output, "float d_output[%d][%d][%d][%d]", batch_size, in_channels, input_height, input_width);
  sprintf(v1, "float v1[%d][%d][%d][%d]", batch_size, in_channels, input_height, input_width);
  sprintf(v2, "float v2[%d]", in_channels);
  sprintf(gamma_and_beta, "float gamma_and_beta[%d]", in_channels * 2);
  sprintf(dv1, "float dv1[%d][%d][%d][%d]", batch_size, in_channels, input_height, input_width);
  sprintf(dv2, "float dv2[%d]", in_channels);
  sprintf(dvar, "float dvar[%d]", in_channels);
  sprintf(dmu, "float dmu[%d]", in_channels);
  sprintf(d_input, "float d_input[%d][%d][%d][%d]", batch_size, in_channels, input_height, input_width);
  sprintf(d_gamma_and_beta, "float d_gamma_and_beta[%d]", in_channels * 2);

  string first_line = "__global__ void backwardPass_BatchNorm2d_l" + to_string(kernel_id) + "(" + d_output + ", " + v1 + ", " + v2 + ", " + gamma_and_beta + ", " + dv1 + ", " + dv2 + ", " + dvar + ", " + dmu + ", " + d_input + ", " + d_gamma_and_beta + ") { \n";

  string codegen = first_line;
  codegen += "  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;\n";
  codegen += "  const long parallel_size = blockDim.x * gridDim.x;\n";
  codegen += "\n";
  codegen += "  const long Z1 = " + to_string(in_channels) + "L;  // C\n";
  codegen += "\n";
  codegen += "  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {\n";
  codegen += "    long idx = n;\n";
  codegen += "    const long i1 = ((idx /= 1) % " + to_string(in_channels) + ");  // c\n";
  codegen += "\n";
  codegen += "    d_gamma_and_beta[i1] = 0;\n";
  codegen += "    d_gamma_and_beta[" + to_string(in_channels) + " + i1] = 0;     // C + i1\n";
  codegen += "  }\n";
  codegen += "\n";
  codegen += "  __syncthreads();\n";
  codegen += "\n";
  codegen += "  const long N1 = " + to_string(batch_size) + "L*" + to_string(in_channels) + "*" + to_string(input_height) + "*" + to_string(input_width) + ";  // N * C * H * W\n";
  codegen += "  const long N2 = " + to_string(in_channels) + "L; // C\n";
  codegen += "\n";
  codegen += "  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {\n";
  codegen += "    long idx = n;\n";
  codegen += "    const long i1 = ((idx /= 1) % " + to_string(batch_size) + "); // n\n";
  codegen += "    const long i2 = ((idx /= " + to_string(batch_size) + ") % " + to_string(in_channels) + "); // c\n";
  codegen += "    const long i3 = ((idx /= " + to_string(in_channels) + ") % " + to_string(input_height) + "); // h\n";
  codegen += "    const long i4 = ((idx /= " + to_string(input_height) + ") % " + to_string(input_width) + "); // w\n";
  codegen += "\n";
  codegen += "    atomicAdd(&d_gamma_and_beta[i2], d_output[i1][i2][i3][i4]);\n";
  codegen += "    atomicAdd(&d_gamma_and_beta[6+i2], (v1[i1][i2][i3][i4]/v2[i2]) * d_output[i1][i2][i3][i4]);\n";
  codegen += "    \n";
  codegen += "  }\n";
  codegen += "  __syncthreads();\n";
  codegen += "\n";
  codegen += "  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {\n";
  codegen += "    long idx = n;\n";
  codegen += "    const long i1 = ((idx /= 1) % " + to_string(batch_size) + "); // n\n";
  codegen += "    const long i2 = ((idx /= " + to_string(batch_size) + ") % " + to_string(in_channels) + "); // c\n";
  codegen += "    const long i3 = ((idx /= " + to_string(in_channels) + ") % " + to_string(input_height) + "); // h\n";
  codegen += "    const long i4 = ((idx /= " + to_string(input_height) + ") % " + to_string(input_width) + "); // w\n";
  codegen += "\n";
  codegen += "    float dx_bar = gamma_and_beta[i2] * d_output[i1][i2][i3][i4];\n";
  codegen += "\n";
  codegen += "    dv1[i1][i2][i3][i4] = (dx_bar) / v2[i2];\n";
  codegen += "    atomicAdd(&dv2[i2], (-1) * dx_bar / (v2[i2]*v2[i2]));\n";
  codegen += "  }\n";
  codegen += "\n";
  codegen += "  __syncthreads();\n";
  codegen += "\n";
  codegen += "  for (long n = N2 * thread_pos / parallel_size; n < N2 * (thread_pos+1) / parallel_size; ++n) {\n";
  codegen += "    long idx = n;\n";
  codegen += "    const long i1 = ((idx /= 1) % " + to_string(in_channels) + ");  // c\n";
  codegen += "\n";
  codegen += "    dvar[i1] = dv2[i1] / (2 * v2[i1]);\n";
  codegen += "  }\n";
  codegen += "\n";
  codegen += "  __syncthreads();\n";
  codegen += "  const long N_bn = " + to_string(batch_size) + "L*" + to_string(input_height) + "*" + to_string(input_width) + ";  // N * H * W\n";
  codegen += "\n";
  codegen += "  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {\n";
  codegen += "    long idx = n;\n";
  codegen += "    const long i1 = ((idx /= 1) % " + to_string(batch_size) + "); // n\n";
  codegen += "    const long i2 = ((idx /= " + to_string(batch_size) + ") % " + to_string(in_channels) + "); // c\n";
  codegen += "    const long i3 = ((idx /= " + to_string(in_channels) + ") % " + to_string(input_height) + "); // h\n";
  codegen += "    const long i4 = ((idx /= " + to_string(input_height) + ") % " + to_string(input_width) + "); // w\n";
  codegen += "\n";
  codegen += "    atomicAdd(&dv1[i1][i2][i3][i4], 2 * v1[i1][i2][i3][i4] * dvar[i2] / N_bn);\n";
  codegen += "  }\n";
  codegen += "\n";
  codegen += "  __syncthreads();\n";
  codegen += "\n";
  codegen += "  \n";
  codegen += "\n";
  codegen += "  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n){\n";
  codegen += "    long idx = n;\n";
  codegen += "    const long i1 = ((idx /= 1) % " + to_string(batch_size) + "); // n\n";
  codegen += "    const long i2 = ((idx /= " + to_string(batch_size) + ") % " + to_string(in_channels) + "); // c\n";
  codegen += "    const long i3 = ((idx /= " + to_string(in_channels) + ") % " + to_string(input_height) + "); // h\n";
  codegen += "    const long i4 = ((idx /= " + to_string(input_height) + ") % " + to_string(input_width) + "); // w\n";
  codegen += "\n";
  codegen += "    atomicAdd(&dmu[i2], -dv1[i1][i2][i3][i4]);\n";
  codegen += "  }\n";
  codegen += "\n";
  codegen += "  __syncthreads();\n";
  codegen += "\n";
  codegen += "  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {\n";
  codegen += "    long idx = n;\n";
  codegen += "    const long i1 = ((idx /= 1) % " + to_string(batch_size) + "); // n\n";
  codegen += "    const long i2 = ((idx /= " + to_string(batch_size) + ") % " + to_string(in_channels) + "); // c\n";
  codegen += "    const long i3 = ((idx /= " + to_string(in_channels) + ") % " + to_string(input_height) + "); // h\n";
  codegen += "    const long i4 = ((idx /= " + to_string(input_height) + ") % " + to_string(input_width) + "); // w\n";
  codegen += "\n";
  codegen += "    d_input[i1][i2][i3][i4] = dv1[i1][i2][i3][i4] + dmu[i2] / N_bn;\n";
  codegen += "  }\n";

  codegen += "}\n";
  return codegen;
}

/*
//(N,C,H,W) = (64, 6, 24, 24), kernel_id = 5, eps = 0.00001, momentum =0.1
//Requirement: parallel_size should ~= C,  must devide N*C*H*W
__global__ void forwardPass_BatchNorm2d_l5(float input[64][6][24][24], float gamma_and_beta[12], float running_m[6], float running_v[6], float mu[6], float var[6], float v1[64][6][24][24], float v2[6], float output[64][6][24][24]){

  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 6;  //C

	for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1	) % 6);  //c

		mu[i1] = 0;
    var[i1] = 0;
	}

  __syncthreads();


	const long N1 = 64*6*24*24;  //N *C * H * W

	for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1	) % 64);  //n
		const long i2 = ((idx /= 64	) % 6);   //c
		const long i3 = ((idx /= 6	) % 24);   //h
		const long i4 = ((idx /= 24	) % 24);   //w

		atomicAdd(&mu[i2], input[i1][i2][i3][i4]);

	}

  __syncthreads();


  const long N_bn = 64*24*24;  //N*H*W
  const long N2 = 6; //C


  for (long n = 0; n < N2; n += parallel_size){
    long idx = n + thread_pos;
    if(idx < N2){
      mu[idx] = mu[idx] / N_bn;
    }
  }

  __syncthreads();
  //We get mu

  //IF tracking running status:


  for (long n = 0; n < N2; n += parallel_size) {
		long idx = n + thread_pos;
		if(idx < N2){
		  running_m[idx] = 0.1*running_m[idx] + 0.9*mu[idx];    // running_m = momentum * running_m + (1-momentum) * mu;
    }
	}

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1	) % 64);  //n
		const long i2 = ((idx /= 64	) % 6);   //c
		const long i3 = ((idx /= 6	) % 24);   //h
		const long i4 = ((idx /= 24	) % 24);   //w

		v1[i1][i2][i3][i4] = input[i1][i2][i3][i4] - running_m[i2];

	}

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1	) % 64);  //n
		const long i2 = ((idx /= 64	) % 6);   //c
		const long i3 = ((idx /= 6	) % 24);   //h
		const long i4 = ((idx /= 24	) % 24);   //w

		atomicAdd(&var[i2], (v1[i1][i2][i3][i4]*v1[i1][i2][i3][i4]));

	}

  __syncthreads();


  for (long n = 0; n < N2; n += parallel_size) {
		long idx = n + thread_pos;
    if(idx < N2){
      var[idx] = var[idx] / N_bn;
    }
	}
  //We get var

  //IF tracking running status:

  __syncthreads();

  for (long n = 0; n < N2; n += parallel_size) {
		long idx = n + thread_pos;
    if(idx < N2){
      running_v[idx] = 0.1*running_v[idx] + 0.9*var[idx];   // running_v = momentum * running_v + (1-momentum) * var;
    }
	}

  __syncthreads();


  for (long n = N2 * thread_pos / parallel_size; n < N2 * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1	) % 6);  //c

		v2[i1] = sqrt(running_v[i1] + 0.00001); //eps = 0.00001, v2 = sqrt(running_v + eps)
	}

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1	) % 64);  //n
		const long i2 = ((idx /= 64	) % 6);   //c
		const long i3 = ((idx /= 6	) % 24);   //h
		const long i4 = ((idx /= 24	) % 24);   //w

		output[i1][i2][i3][i4] = (v1[i1][i2][i3][i4] / v2[i2]) * gamma_and_beta[i2] + gamma_and_beta[6 + i2];    //gamma = gamma_and_beta[0~c-1], beta = gamma_and_beta[c~2c-1]; output = (v1/v2) * gamma + beta
	}

}




//(N,C,H,W) = (64, 6, 24, 24), kernel_id = 5, eps = 0.00001, momentum =0.1
//Requirement: parallel_size should ~= C,  must devide N*C*H*W
__global__ void backwardPass_BatchNorm2d_l5(float d_output[64][6][24][24], float v1[64][6][24][24], float v2[6], float gamma_and_beta[12], float dv1[64][6][24][24], float dv2[6], float dvar[6], float dmu[6], float d_input[64][6][24][24], float d_gamma_and_beta[12]){

  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 6;  //C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1	) % 6);  //c

		d_gamma_and_beta[i1] = 0;
    d_gamma_and_beta[6+i1] = 0;     //C+i1
	}

  __syncthreads();

  const long N1 = 64*6*24*24;   //N*C*H*W
  const long N2 = 6; //C

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n){
    long idx = n;
    const long i1 = ((idx /= 1	) % 64);  //n
		const long i2 = ((idx /= 64	) % 6);   //c
		const long i3 = ((idx /= 6	) % 24);   //h
		const long i4 = ((idx /= 24	) % 24);   //w

    atomicAdd(&d_gamma_and_beta[i2], d_output[i1][i2][i3][i4]);
    atomicAdd(&d_gamma_and_beta[6+i2], (v1[i1][i2][i3][i4]/v2[i2]) * d_output[i1][i2][i3][i4]);

  }
  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n){
    long idx = n;
    const long i1 = ((idx /= 1	) % 64);  //n
		const long i2 = ((idx /= 64	) % 6);   //c
		const long i3 = ((idx /= 6	) % 24);   //h
		const long i4 = ((idx /= 24	) % 24);   //w

    float dx_bar = gamma_and_beta[i2] * d_output[i1][i2][i3][i4];

    dv1[i1][i2][i3][i4] = (dx_bar) / v2[i2];
    atomicAdd(&dv2[i2], (-1) * dx_bar / (v2[i2]*v2[i2]));
  }

  __syncthreads();


  for (long n = N2 * thread_pos / parallel_size; n < N2 * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1	) % 6);  //c

    dvar[i1] = dv2[i1] / (2 * v2[i1]);
	}

  __syncthreads();
  const long N_bn = 64*24*24;  //N*H*W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n){
    long idx = n;
    const long i1 = ((idx /= 1	) % 64);  //n
		const long i2 = ((idx /= 64	) % 6);   //c
		const long i3 = ((idx /= 6	) % 24);   //h
		const long i4 = ((idx /= 24	) % 24);   //w

    atomicAdd(&dv1[i1][i2][i3][i4], 2 * v1[i1][i2][i3][i4] * dvar[i2] / N_bn);
  }

  __syncthreads();



  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n){
    long idx = n;
    const long i1 = ((idx /= 1	) % 64);  //n
		const long i2 = ((idx /= 64	) % 6);   //c
		const long i3 = ((idx /= 6	) % 24);   //h
		const long i4 = ((idx /= 24	) % 24);   //w

    atomicAdd(&dmu[i2], -dv1[i1][i2][i3][i4]);
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n){
    long idx = n;
    const long i1 = ((idx /= 1	) % 64);  //n
		const long i2 = ((idx /= 64	) % 6);   //c
		const long i3 = ((idx /= 6	) % 24);   //h
		const long i4 = ((idx /= 24	) % 24);   //w

    d_input[i1][i2][i3][i4] = dv1[i1][i2][i3][i4] + dmu[i2] / N_bn;
  }
}


*/

string add_forward_CodeGen(int kernel_id, int num_inputs, int N) {
  string codegen;
  codegen += "__global__ void forwardPass_Add_l" + to_string(kernel_id) + "(";
  for (long i = 0; i < num_inputs; i++)
    codegen += "float* input" + to_string(i) + ", ";
  codegen += "float* out, long N) {\n";
  codegen += "  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;\n";
  codegen += "  const long parallel_size = blockDim.x * gridDim.x;\n";
  codegen += "\n";
  codegen += "  for(long n = 0; n < N; n += parallel_size) {\n";
  codegen += "    long idx = n + thread_pos;\n";
  codegen += "      if (idx < N) {\n";
  codegen += "        out[idx] = ";
  for (long i = 0; i < num_inputs; i++)
    codegen += "input" + to_string(i) + "[idx] + ";
  codegen.erase(codegen.size() - 3);
  codegen += ";\n";
  codegen += "      }\n";
  codegen += "  }\n";
  codegen += "}\n";
  codegen += "\n";

  return codegen;
}

// __global__ void forwardPass_Add_l5(float* input1, float* input2, float* out, long N) {
//   const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
//   const long parallel_size = blockDim.x * gridDim.x;

//   for(long n = 0; n < N; n += parallel_size) {
//     long idx = n + thread_pos;
//       if (idx < N) {
//         out[idx] = input1[idx] + input2[idx];
//       }
//   }
// }

string multiGradient_add_CodeGen(int kernel_id, int num_inputs, int N) {
  string codegen;
  codegen += "__global__ void multiGradient_Add_l" + to_string(kernel_id) + "(";
  for (long i = 0; i < num_inputs; i++)
    codegen += "float* d_output" + to_string(i) + ", ";
  codegen += "long N) {\n";
  codegen += "  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;\n";
  codegen += "  const long parallel_size = blockDim.x * gridDim.x;\n";
  codegen += "\n";
  codegen += "  for(long n = 0; n < N; n += parallel_size) {\n";
  codegen += "    long idx = n + thread_pos;\n";
  codegen += "      if (idx < N) {\n";
  codegen += "        d_output0[idx] += ";
  for (long i = 0; i < num_inputs; i++)
    codegen += "d_output" + to_string(i) + "[idx] + ";
  codegen.erase(codegen.size() - 3);
  codegen += ";\n";
  codegen += "      }\n";
  codegen += "  }\n";
  codegen += "}\n";
  codegen += "\n";

  return codegen;
}

// __global__ void multiGradient_Add_l5(float* d_output1, float* d_output2, long N) {
//   const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
//   const long parallel_size = blockDim.x * gridDim.x;

//   for(long n = 0; n < N; n += parallel_size) {
//     long idx = n + thread_pos;
//       if (idx < N) {
//         d_output1[idx] += d_output2[idx];
//       }
//   }
// }

string concat_forward_CodeGen(int kernel_id, int batch_num, vector<int> &input_Cs, int input_height, int input_width) {
  string codegen;

  int num_inputs = input_Cs.size();
  int out_channels = 0;

  codegen += "__global__ void concat_forward_l" + to_string(kernel_id) + "(";
  for (int input_idx = 0; input_idx < input_Cs.size(); input_idx++) {
    int in_channels = input_Cs[input_idx];
    codegen += "float input" + to_string(input_idx) + "[" + to_string(batch_num) + "][" + to_string(in_channels) + "][" + to_string(input_height) + "][" + to_string(input_width) + "], ";
    out_channels += in_channels;
  }
  codegen += "float output[" + to_string(batch_num) + "][" + to_string(out_channels) + "][" + to_string(input_height) + "][" + to_string(input_width) + "]) {\n";

  codegen += "  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;\n";
  codegen += "  const long parallel_size = blockDim.x * gridDim.x;\n";
  codegen += "\n";
  for (int input_idx = 0; input_idx < input_Cs.size() - 1; input_idx++) {
    int in_channels = input_Cs[input_idx];
    codegen += "  long dim" + to_string(input_idx) + " = " + to_string(in_channels) + ";\n";
  }
  codegen += "\n";
  codegen += "  const long N = " + to_string(batch_num) + "L*" + to_string(out_channels) + "*" + to_string(input_height) + "*" + to_string(input_width) + ";  // N * sum(C) * H * W\n";
  codegen += "\n";
  codegen += "  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {\n";
  codegen += "    long idx = n;\n";
  codegen += "    const long i1 = ((idx /= 1) % " + to_string(batch_num) + ");  //n\n";
  codegen += "    const long i2 = ((idx /= " + to_string(batch_num) + ") % " + to_string(out_channels) + ");  //c\n";
  codegen += "    const long i3 = ((idx /= " + to_string(out_channels) + ") % " + to_string(input_height) + ");  //h\n";
  codegen += "    const long i4 = ((idx /= " + to_string(input_height) + ") % " + to_string(input_width) + ");  //w\n";
  codegen += "    if (i2 < dim0) {\n";
  codegen += "      output[i1][i2][i3][i4] = input0[i1][i2][i3][i4];\n";
  for (int input_idx = 1; input_idx < input_Cs.size() - 1; input_idx++) {
    int in_channels = input_Cs[input_idx];
    codegen += "    } else if (i2 < dim0";
    for (int i = 1; i <= input_idx; i++)
      codegen += " + dim" + to_string(i);
    codegen += ") {\n";
    codegen += "      output[i1][i2][i3][i4] = input" + to_string(input_idx) + "[i1][i2 - dim0";
    for (int i = 1; i < input_idx; i++)
      codegen += " - dim" + to_string(i);
    codegen += "][i3][i4];\n";
  }
  codegen += "    } else {\n";
  codegen += "      output[i1][i2][i3][i4] = input" + to_string(input_Cs.size() - 1) + "[i1][i2 - dim0";
  for (int i = 1; i < input_Cs.size() - 1; i++)
    codegen += " - dim" + to_string(i);
  codegen += "][i3][i4];\n";
  codegen += "    }\n";
  codegen += "  }\n";
  codegen += "}\n";
  codegen += "\n";

  return codegen;
}

// __global__ void concat_forward_l5(float[64][6][24][24] input0, float[64][2][24][24] input1, float[64][5][24][24] input2, float[64][13][24][24] output) {
//   const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
//   const long parallel_size = blockDim.x * gridDim.x;

//   long dim0 = 6;
//   long dim1 = 2;
//   long dim2 = 5;

//   const long N = 64 * 13 * 24 * 24; // N * sum(C) * H * W

//   for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
// 		long idx = n;
// 		const long i1 = ((idx /= 1	) % 64);  //n
// 		const long i2 = ((idx /= 64	) % 13);  //sum(c)
// 		const long i3 = ((idx /= 13 ) % 24);  //h
// 		const long i4 = ((idx /= 24	) % 24);  //w
//     if (i2 < dim0) {
//       output[i1][i2][i3][i4] = input0[i1][i2][i3][i4];
//     } else if (i2 < dim0 + dim1) {
//       output[i1][i2][i3][i4] = input1[i1][i2 - dim0][i3][i4];
//     } else if (i2 < dim0 + dim1 + dim2) {
//       output[i1][i2][i3][i4] = input2[i1][i2 - dim0 - dim1][i3][i4];
//     }
//   }
// }

string concat_backward_CodeGen(int kernel_id, int batch_num, vector<int> &input_Cs, int input_height, int input_width) {
  string codegen;

  int num_inputs = input_Cs.size();
  int out_channels = 0;

  codegen += "__global__ void concat_backward_l" + to_string(kernel_id) + "(";
  for (int input_idx = 0; input_idx < input_Cs.size(); input_idx++) {
    int in_channels = input_Cs[input_idx];
    codegen += "float d_input" + to_string(input_idx) + "[" + to_string(batch_num) + "][" + to_string(in_channels) + "][" + to_string(input_height) + "][" + to_string(input_width) + "], ";
    out_channels += in_channels;
  }
  codegen += "float d_output[" + to_string(batch_num) + "][" + to_string(out_channels) + "][" + to_string(input_height) + "][" + to_string(input_width) + "]) {\n";

  codegen += "  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;\n";
  codegen += "  const long parallel_size = blockDim.x * gridDim.x;\n";
  codegen += "\n";
  for (int input_idx = 0; input_idx < input_Cs.size() - 1; input_idx++) {
    int in_channels = input_Cs[input_idx];
    codegen += "  long dim" + to_string(input_idx) + " = " + to_string(in_channels) + ";\n";
  }
  codegen += "\n";
  codegen += "  const long N = " + to_string(batch_num) + "L*" + to_string(out_channels) + "*" + to_string(input_height) + "*" + to_string(input_width) + ";  // N * sum(C) * H * W\n";
  codegen += "\n";
  codegen += "  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {\n";
  codegen += "		long idx = n;\n";
  codegen += "    const long i1 = ((idx /= 1) % " + to_string(batch_num) + ");  //n\n";
  codegen += "    const long i2 = ((idx /= " + to_string(batch_num) + ") % " + to_string(out_channels) + ");  //c\n";
  codegen += "    const long i3 = ((idx /= " + to_string(out_channels) + ") % " + to_string(input_height) + ");  //h\n";
  codegen += "    const long i4 = ((idx /= " + to_string(input_height) + ") % " + to_string(input_width) + ");  //w\n";
  codegen += "    if (i2 < dim0) {\n";
  codegen += "      d_input0[i1][i2][i3][i4] = d_output[i1][i2][i3][i4];\n";
  for (int input_idx = 1; input_idx < input_Cs.size() - 1; input_idx++) {
    int in_channels = input_Cs[input_idx];
    codegen += "    } else if (i2 < dim0";
    for (int i = 1; i <= input_idx; i++)
      codegen += " + dim" + to_string(i);
    codegen += ") {\n";
    codegen += "      d_input" + to_string(input_idx) + "[i1][i2 - dim0";
    for (int i = 1; i < input_idx; i++)
      codegen += " - dim" + to_string(i);
    codegen += "][i3][i4] = d_output[i1][i2][i3][i4];\n";
  }
  codegen += "    } else {\n";
  codegen += "      d_input" + to_string(input_Cs.size() - 1) + "[i1][i2 - dim0";
  for (int i = 1; i < input_Cs.size() - 1; i++)
    codegen += " - dim" + to_string(i);
  codegen += "][i3][i4] = d_output[i1][i2][i3][i4];\n";
  codegen += "    }\n";
  codegen += "  }\n";
  codegen += "}\n";
  codegen += "\n";

  return codegen;
}

// __global__ void concat_backward_l5(float[64][6][24][24] d_input1, float[64][2][24][24] d_input2, float[64][5][24][24] d_input3, float[64][13][24][24] d_output) {
//   const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
//   const long parallel_size = blockDim.x * gridDim.x;

//   long dim1 = 6;
//   long dim2 = 2;
//   long dim2 = 5;

//   const long N = 64 * 8 * 24 * 24; // N * sum(C) * H * W

//   for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
// 		long idx = n;
// 		const long i1 = ((idx /= 1	) % 64);  //n
// 		const long i2 = ((idx /= 64	) % 13);  //sum(c)
// 		const long i3 = ((idx /= 13 ) % 24);  //h
// 		const long i4 = ((idx /= 24	) % 24);  //w
//     if (i2 < dim1) {
//       d_input0[i1][i2][i3][i4] = d_output[i1][i2][i3][i4];
//     } else if (i2 < dim1 + dim2) {
//       d_input1[i1][i2 - dim1][i3][i4] = d_output[i1][i2][i3][i4];
//     } else if (i2 < dim1 + dim2 + dim3) {
//       d_input2[i1][i2 - dim1 - dim2][i3][i4] = d_output[i1][i2][i3][i4];
//     }
//   }
// }

string scale_forward_CodeGen(int kernel_id, int batch_num, int in_channel, int scale_height, int scale_width) {
  string codegen;

  codegen += "__global__ void scale_forward_l" + to_string(kernel_id);
  codegen += "(float input[" + to_string(batch_num) + "][" + to_string(in_channel) + "][1][1], float output[" + to_string(batch_num) + "][" + to_string(in_channel) + "][" + to_string(scale_height) + "][" + to_string(scale_width) + "]) {\n";
  codegen += "  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;\n";
  codegen += "  const long parallel_size = blockDim.x * gridDim.x;\n";
  codegen += "  const long N = " + to_string(batch_num) + "L*" + to_string(in_channel) + "*" + to_string(scale_height) + "*" + to_string(scale_width) + ";  // N * C * scale_H * scale_W\n";
  codegen += "  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {\n";
  codegen += "		long idx = n;\n";
  codegen += "    const long i1 = ((idx /= 1) % " + to_string(batch_num) + ");  //n\n";
  codegen += "    const long i2 = ((idx /= " + to_string(batch_num) + ") % " + to_string(in_channel) + ");  //c\n";
  codegen += "    const long i3 = ((idx /= " + to_string(in_channel) + ") % " + to_string(scale_height) + ");  //h\n";
  codegen += "    const long i4 = ((idx /= " + to_string(scale_height) + ") % " + to_string(scale_width) + ");  //w\n";
  codegen += "    output[i1][i2][i3][i4] = input[i1][i2][0][0];\n";
  codegen += "  }\n";
  codegen += "}\n";
  codegen += "\n";

  return codegen;
}

// __global__ void scale_forward_l5(float[64][6][1][1] input, float[64][6][24][24] output) {
//   const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
//   const long parallel_size = blockDim.x * gridDim.x;

//   const long N = 64 * 6 * 24 * 24; // N * C * scale_H * scale_W

//   for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
// 		long idx = n;
// 		const long i1 = ((idx /= 1) % 64);   //n
// 		const long i2 = ((idx /= 64) % 8);   //c
// 		const long i3 = ((idx /= 8) % 24);   //h
// 		const long i4 = ((idx /= 24) % 24);  //w
//     output[i1][i2][i3][i4] = input[i1][i2][0][0];
//   }
// }

string scale_backward_CodeGen(int kernel_id, int batch_num, int in_channel, int scale_height, int scale_width) {
  string codegen;

  codegen += "__global__ void scale_backward_l" + to_string(kernel_id);
  codegen += "(float d_input[" + to_string(batch_num) + "][" + to_string(in_channel) + "][1][1], float d_output[" + to_string(batch_num) + "][" + to_string(in_channel) + "][" + to_string(scale_height) + "][" + to_string(scale_width) + "]) {\n";
  codegen += "  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;\n";
  codegen += "  const long parallel_size = blockDim.x * gridDim.x;\n";
  codegen += "  const long N = " + to_string(batch_num) + "L*" + to_string(in_channel) + "*" + to_string(scale_height) + "*" + to_string(scale_width) + ";  // N * C * scale_H * scale_W\n";
  codegen += "  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {\n";
  codegen += "		long idx = n;\n";
  codegen += "    const long i1 = ((idx /= 1) % " + to_string(batch_num) + ");  //n\n";
  codegen += "    const long i2 = ((idx /= " + to_string(batch_num) + ") % " + to_string(in_channel) + ");  //c\n";
  codegen += "    const long i3 = ((idx /= " + to_string(in_channel) + ") % " + to_string(scale_height) + ");  //h\n";
  codegen += "    const long i4 = ((idx /= " + to_string(scale_height) + ") % " + to_string(scale_width) + ");  //w\n";
  codegen += "    atomicAdd(&d_input[i1][i2][0][0], d_output[i1][i2][i3][i4] / " + to_string(scale_height * scale_width) + ");\n";
  codegen += "  }\n";
  codegen += "}\n";
  codegen += "\n";

  return codegen;
}

// __global__ void scale_backward_l5(float[64][6][1][1] d_input, float[64][6][24][24] d_output) {
//   const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
//   const long parallel_size = blockDim.x * gridDim.x;

//   const long N = 64 * 6 * 24 * 24; // N * C * scale_H * scale_W

//   for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
//     long idx = n;
//     const long i1 = ((idx /= 1) % 64);   //n
//     const long i2 = ((idx /= 64) % 8);  //c
//     const long i3 = ((idx /= 8) % 24);   //h
//     const long i4 = ((idx /= 24) % 24); //w
//     atomicAdd(&d_input[i1][i2][0][0], d_output[i1][i2][i3][i4] / (scale_H * scale_W));
//   }
// }

extern vector<Tensor*> tensor_list;
extern vector<CUDAKernel> kernel_list;
extern vector<Model_Layer*> forward_layers;

extern long long memory_offset_intermediate;
extern long long memory_offset_weights;

/**
 * @brief get the dimension used in casting of the input elements using regex expression
 *
 * @param input generated function declaration
 * @param output vector of dimensions
 * @return number of these dimensions captured
 */
static int get_dims(string &input, vector<string> &output) {
    string to_search = input.substr(0, input.find('\n'));
    output.clear();
    regex pattern("\\[[0-9]+\\](\\[[0-9\\[\\]]+\\])*");
    smatch matches;
    while (regex_search(to_search, matches, pattern)) {
        output.push_back(matches.str(1));
        to_search = matches.suffix();
    }
    return output.size();
}

/**
 * @brief get the complete dimension of the input elements using regex expression
 *
 * @param input generated function declaration
 * @param output vector of dimensions
 * @return number of these dimensions captured
 */
static int get_raw_dims(string &input, vector<string> &output) {
    string to_search = input.substr(0, input.find('\n'));
    output.clear();
    regex pattern("\\[[0-9\\[\\]]+\\]");
    smatch matches;
    while (regex_search(to_search, matches, pattern)) {
        output.push_back(matches.str());
        to_search = matches.suffix();
    }
    return output.size();
}

/**
 * @brief get the dimension of the input elements using regex expression
 *
 * @param input generated function declaration
 * @param output vector of dimensions
 * @return number of these dimensions captured
 */
static int get_func_headers(string input, vector<string> &output) {
    output.clear();
    regex pattern("__global__ void [A-z0-9_]+\\(.*\\)");
    smatch matches;
    while (regex_search(input, matches, pattern)) {
        output.push_back(matches.str());
        input = matches.suffix();
    }
    return output.size();
}

/**
 * @brief get the file relative path
 * 
 * @param file_idx file index, used to specify
 * @param header if the file is a header
 * @return the relative path to the file 
 */
static string get_file(int file_idx, bool header) {
    if (header)
        return "include/declaration" + to_string(file_idx) + ".cuh";
    return "src/declaration" + to_string(file_idx) + ".cu";
}

/**
 * @brief determine if a kernel type need special declaration or not (i.e. whether the declaration 
 *        is in "cudannUtils.cuh") 
 * 
 * @param type type of cuda kernel
 * @return if the type of kernel specified need to be declared
 */
static bool need_specific_declaration(CUDAKernelType type) {
    switch (type) {
        case Conv2d_Forward:
        case ReLU_Forward:
        case MaxPool2d_Forward:
        case AdaptiveAvgPool2d_Forward:
        case Linear_Forward:
        case BatchNorm2d_Forward:
        case Add_Forward:
        case Concat_Forward:
        case Scale_Forward:
        case Conv2d_Backward_Weight:
        case Conv2d_Backward_Input:
        case ReLU_Backward:
        case MaxPool2d_Backward:
        case AdaptiveAvgPool2d_Backward:
        case Linear_Backward_Weight:
        case Linear_Backward_Input:
        case Linear_Backward_Bias:
        case BatchNorm2d_Backward:
        case Add_MultiGredient:
        case Concat_Backward:
        case Scale_Backward:
            return true;
        case Dropout_Forward:
        case Conv2d_Apply_Grad:
        case Linear_Apply_Grad_Bias:
        case Linear_Apply_Grad_Weight:
        case BatchNorm2d_Apply_Grad:
        case Dropout_Backward:
        case LoadData_A0:
        case makeLoss:
            return false;
        default:
            Assert(false);
    }
    return false;
}

/**
 * @brief generate tensor declarations according to the number of tensors
 * 
 * @param tensors the vector of tensors tat need to be declared
 * @param num_tensor_per_line number of tensor declarations per line
 * @return all tensor declaration string
 */
static string generate_tensor_declarations(vector<Tensor *> &tensors,
                                           int num_tensor_per_line) {
    string tensor_declaration_gen = "// declare all tensors\n";
    string to_add;
    for (int index = 0; index <= tensors.size(); index++) {
        if (to_add.size() != 0 &&
                (index % num_tensor_per_line == 0 ||
                 index == tensors.size())) {
            to_add.erase(to_add.length() - 2);
            tensor_declaration_gen += "float " + to_add + ";\n";
            to_add.clear();
        }
        if (index == tensors.size())
            break;
        Tensor *tensor = tensors[index];
        to_add += "*" + tensor->name() + ", ";
    }
    Assert(to_add.size() == 0);
    tensor_declaration_gen += "\n";
    return tensor_declaration_gen;
}

extern bool is_UVM;
/**
 * @brief generate helper function according to the UVM setting
 */
static string generate_helper_functions() {
    string helper_func_gen;
    if (is_UVM) {
        helper_func_gen += "// helper functions\n";
        helper_func_gen += "void generate_data(float *base, long long size) {\n";
        helper_func_gen += "  for(long long i = 0; i < size; i++){\n";
        helper_func_gen += "    base[i] = (float)(rand() % 256); \n";
        helper_func_gen += "  }\n";
        helper_func_gen += "}\n";
        helper_func_gen += "\n";
        helper_func_gen += "void generate_mask(float *base, long long size, float p) {\n";
        helper_func_gen += "  for(long long i = 0; i < size; i++){\n";
        helper_func_gen += "    float m = (float) (rand() % 1000000); \n";
        helper_func_gen += "    base[i] = m < (p * 1000000) ? 1.0f : 0.0f;\n";
        helper_func_gen += "  }\n";
        helper_func_gen += "}\n";
        helper_func_gen += "\n";
    } else {
        helper_func_gen += "// helper functions\n";
        helper_func_gen += "void generate_data(float *base, long long size) {\n";
        helper_func_gen += "  float *host_data = (float *) malloc(size * 4);\n";
        helper_func_gen += "  for(long long i = 0; i < size; i++){\n";
        helper_func_gen += "    host_data[i] = (float)(rand() % 256); \n";
        helper_func_gen += "  }\n";
        helper_func_gen += "  cudaMemcpy(base, host_data, sizeof(float) * size, cudaMemcpyHostToDevice);\n";
        helper_func_gen += "  free(host_data);\n";
        helper_func_gen += "}\n";
        helper_func_gen += "\n";
        helper_func_gen += "void generate_mask(float *base, long long size, float p) {\n";
        helper_func_gen += "  float *host_data = (float *) malloc(size * 4);\n";
        helper_func_gen += "  for(long long i = 0; i < size; i++){\n";
        helper_func_gen += "    float m = (float) (rand() % 1000000); \n";
        helper_func_gen += "    host_data[i] = m < (p * 1000000) ? 1.0f : 0.0f;\n";
        helper_func_gen += "  }\n";
        helper_func_gen += "  cudaMemcpy(base, host_data, sizeof(float) * size, cudaMemcpyHostToDevice);\n";
        helper_func_gen += "  free(host_data);\n";
        helper_func_gen += "}\n";
        helper_func_gen += "\n";
    }
    return helper_func_gen;
}

// metadata to be used in main code generation
extern int num_iteration;
int num_threads = 0;
bool is_individual = 1;
extern string output_folder_name;

/**
 * @brief generate all the code according to the model and flags specified by the user
 */
void main_code_generation() {
    iprintf("\nCodegen start\n", "");

    // generated function definition
    string definition_file_gen;

    // includes
    string cu_includes_gen;
    // beginning of main/profile file
    string file_begin_gen;
    // helper functions
    string helper_func_gen;
    // declaration of all tensors
    string tensor_declaration_gen;
    // run iteration function
    string run_iter_gen;
    // main function before malloc
    string main_begin_gen;
    string main_malloc_gen;
    string main_addr_assign_gen;
    string main_data_gen_gen;
    string main_end_gen;

    // number of kernel declaration/definitions in a file, can be customized
    const int kernels_per_file = 50;
    // number of tensor declaration in a line, can be customized
    const int num_tensor_per_line = 8;


    int total_kernel_num = 0;
    for (CUDAKernel kernel : kernel_list)
        if (need_specific_declaration(kernel.type))
            total_kernel_num++;
    int file_num = (int) ceil((double) total_kernel_num / kernels_per_file);

    // includes
    cu_includes_gen += "#include \"include/cudadnnUtil.cuh\"\n";
    for (int file_idx = 0; file_idx < file_num; file_idx++)
        cu_includes_gen += "#include \"" + get_file(file_idx, true) + "\"\n";

    file_begin_gen += "#include <cuda.h>\n";
    file_begin_gen += "#include <cstdio>\n";
    file_begin_gen += "#include <stdlib.h>\n";
    file_begin_gen += "#include <time.h>\n";
    file_begin_gen += "\n";

    file_begin_gen += "// Error checking marco\n";
    file_begin_gen += "#define CUDA_CHECK_RET(x)                                               \\\n";
    file_begin_gen += "  do {                                                                  \\\n";
    file_begin_gen += "    cudaError_t err_code = x;                                           \\\n";
    file_begin_gen += "    if (err_code != 0) {                                                \\\n";
    // use "\045" to represent/escape "%"
    file_begin_gen += "      printf(\"Failed at \045s:\045d w/ code \045d <\045s>, abort\\n\",                \\\n";
    file_begin_gen += "          __FILE__, __LINE__, err_code, cudaGetErrorString(err_code));  \\\n";
    file_begin_gen += "      exit(1);                                                          \\\n";
    file_begin_gen += "    }                                                                   \\\n";
    file_begin_gen += "  } while (0)\n";
    file_begin_gen += "\n";

    // global variables
    printf("Weight size total:       %10lld B\n", memory_offset_weights);
    printf("Intermediate size total: %10lld B\n", memory_offset_intermediate);
    if (!is_individual) {
        file_begin_gen += "// Unified pointer declaration\n";
        file_begin_gen += "float *dataset_base;\n";
        file_begin_gen += "float *weight_base;\n";
        file_begin_gen += "float *intermediate_base;\n";
        file_begin_gen += "\n";

        file_begin_gen += "// Some constants\n";
        file_begin_gen += "const long long weight_size_in_byte = " + to_string(memory_offset_weights) + "LL;\n";
        file_begin_gen += "const long long intermediate_size_in_byte = " + to_string(memory_offset_intermediate) + "LL;\n";
        file_begin_gen += "const int num_iteration = " + to_string(num_iteration) + ";\n";
        file_begin_gen += "\n";
    } else {
        file_begin_gen += "// Workspace\n";
        file_begin_gen += "float *workspace_base;\n";
        file_begin_gen += "\n";
    }

    // kernel function declaration
    map<int, string> kernel_definition_filename;
    int reverse_mapping_idx = 1; // transfer_A0 not compiled now

    int file_idx = 0;
    int kernel_in_file_idx = 0;
    long A0_size = -1;
    vector<string> headers;
    // one extra loop count for capturing and saving remaining contents in the headers
    // that does not fill a complete bundle
    for (int kernel_index = 0; kernel_index <= kernel_list.size(); kernel_index++) {
        if (definition_file_gen.size() != 0 &&
              (kernel_in_file_idx % kernels_per_file == 0 ||
              kernel_index == kernel_list.size())) {
            // get function headers (declarations)
            get_func_headers(definition_file_gen, headers);

            // header file inclusion guard
            string prefix, postfix;
            prefix += "#ifndef __DECLARACTION" + to_string(file_idx) + "_H__\n";
            prefix += "#define __DECLARACTION" + to_string(file_idx) + "_H__\n\n";
            prefix += "#include \"../" + get_file(file_idx, true) + "\"\n\n";
            postfix += "#endif\n";

            ofstream fout;
            string file;

            // header file, write declarations
            file = "./" + output_folder_name + "/" + get_file(file_idx, true);
            fout.open(file, ofstream::out);
            fout << prefix;
            for (string header : headers) {
                fout << header << ";\n\n";
                // skip kernels that does not declaration
                while (!need_specific_declaration(kernel_list[reverse_mapping_idx].type))
                  reverse_mapping_idx++;
                kernel_definition_filename[reverse_mapping_idx++] =
                    "#include \"../" + get_file(file_idx, true) + "\"";
            }
            fout << postfix;
            fout.close();

            // source file, write definitions
            file = "./" + output_folder_name + "/" + get_file(file_idx, false);
            fout.open(file, ofstream::out);
            fout << definition_file_gen;
            fout.close();

            definition_file_gen.clear();
            file_idx++;
        }
        // leave the loop after final bundle completed
        if (kernel_index == kernel_list.size())
            break;

        CUDAKernel kernel = kernel_list[kernel_index];
        switch (kernel.type) {
            case Conv2d_Forward: {
                Conv2d *layer = dynamic_cast<Conv2d *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);
                definition_file_gen += conv2d_forward_CodeGen(kernel.parent_layer->layer_id, 
                        kernel.parent_layer->H, kernel.parent_layer->W, 
                        kernel.parent_layer->N, kernel.parent_layer->C, 
                        layer->out_channels, layer->kernel_size_r, layer->kernel_size_s, 
                        layer->stride_0, layer->stride_1, layer->padding_0, layer->padding_1, layer->bias) + "\n";
                break;
            }
            case ReLU_Forward: {
                ReLU *layer = dynamic_cast<ReLU *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);
                definition_file_gen += reLU_forward_CodeGen(kernel.parent_layer->layer_id, 
                        kernel.parent_layer->N, kernel.parent_layer->C, 
                        kernel.parent_layer->H, kernel.parent_layer->W) + "\n";
                break;
            }
            case MaxPool2d_Forward: {
                MaxPool2d *layer = dynamic_cast<MaxPool2d *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);
                definition_file_gen += maxPool2d_forward_CodeGen(kernel.parent_layer->layer_id, 
                        kernel.parent_layer->N, kernel.parent_layer->C, 
                        kernel.parent_layer->H, kernel.parent_layer->W, 
                        layer->kernel_size, layer->kernel_size, layer->stride, 
                        layer->padding, layer->dilation, layer->ceil_mode) + "\n";
                break;
            }
            case AdaptiveAvgPool2d_Forward: {
                AdaptiveAvgPool2d *layer = dynamic_cast<AdaptiveAvgPool2d *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);
                definition_file_gen += avgAdaptivePool2d_forward_CodeGen(kernel.parent_layer->layer_id, 
                        kernel.parent_layer->N, kernel.parent_layer->C, 
                        kernel.parent_layer->H, kernel.parent_layer->W, 
                        layer->outputsize_0, layer->outputsize_1) + "\n";
                break;
            }
            case Linear_Forward: {
                Linear *layer = dynamic_cast<Linear *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);
                definition_file_gen += linear_forward_CodeGen(kernel.parent_layer->layer_id, 
                        kernel.parent_layer->N, kernel.parent_layer->C, 
                        kernel.parent_layer->H, kernel.parent_layer->W, 
                        layer->in_features, layer->out_features) + "\n";
                break;
            }
            case BatchNorm2d_Forward: {
                BatchNorm2d *layer = dynamic_cast<BatchNorm2d *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);
                definition_file_gen += batchnorm2d_forward_CodeGen(kernel.parent_layer->layer_id, 
                        kernel.parent_layer->N, kernel.parent_layer->C, 
                        kernel.parent_layer->H, kernel.parent_layer->W, 
                        layer->eps, layer->momentum, layer->track_running_stats) + "\n";
                break;
            }
            case Add_Forward: {
                AddOp *layer = dynamic_cast<AddOp *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);
                Assert(kernel.parent_layer->other_inputs.size() > 0);
                definition_file_gen += add_forward_CodeGen(kernel.parent_layer->layer_id, 
                        kernel.parent_layer->other_inputs.size() + 1, 
                        kernel.parent_layer->output_activation->size_in_byte / sizeof(float)) + "\n";
                break;
            }
            case Concat_Forward: {
                ConcatOp *layer = dynamic_cast<ConcatOp *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);
                definition_file_gen += concat_forward_CodeGen(kernel.parent_layer->layer_id, 
                        kernel.parent_layer->N, kernel.parent_layer->input_Cs, 
                        kernel.parent_layer->H, kernel.parent_layer->W) + "\n";
                break;
            }
            case Scale_Forward: {
                ScaleOp *layer = dynamic_cast<ScaleOp *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);
                definition_file_gen += scale_forward_CodeGen(kernel.parent_layer->layer_id, 
                        kernel.parent_layer->N, kernel.parent_layer->C, 
                        kernel.parent_layer->scale_H, kernel.parent_layer->scale_W) + "\n";
                break;
            }
            case Conv2d_Backward_Weight: {
                Conv2d *layer = dynamic_cast<Conv2d *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);
                definition_file_gen += conv2d_backward_weight_CodeGen(kernel.parent_layer->layer_id, 
                        kernel.parent_layer->H, kernel.parent_layer->W, 
                        kernel.parent_layer->N, kernel.parent_layer->C, 
                        layer->out_channels, layer->kernel_size_r, layer->kernel_size_s, 
                        layer->stride_0, layer->stride_1, layer->padding_0, layer->padding_1, layer->bias) + "\n";
                break;
            }
            case Conv2d_Backward_Input: {
                Conv2d *layer = dynamic_cast<Conv2d *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);
                definition_file_gen += conv2d_backward_input_CodeGen(kernel.parent_layer->layer_id, 
                        kernel.parent_layer->H, kernel.parent_layer->W, 
                        kernel.parent_layer->N, kernel.parent_layer->C, 
                        layer->out_channels, layer->kernel_size_r, layer->kernel_size_s, 
                        layer->stride_0, layer->stride_1, layer->padding_0, layer->padding_1, layer->bias) + "\n";
                break;
            }
            case ReLU_Backward: {
                ReLU *layer = dynamic_cast<ReLU *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);
                definition_file_gen += reLU_backward_CodeGen(kernel.parent_layer->layer_id, 
                        kernel.parent_layer->N, kernel.parent_layer->C, 
                        kernel.parent_layer->H, kernel.parent_layer->W) + "\n";
                break;
            }
            case MaxPool2d_Backward: {
                MaxPool2d *layer = dynamic_cast<MaxPool2d *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);
                definition_file_gen += maxPool2d_backward_CodeGen(kernel.parent_layer->layer_id, 
                        kernel.parent_layer->N, kernel.parent_layer->C, 
                        kernel.parent_layer->H, kernel.parent_layer->W, 
                        layer->kernel_size, layer->kernel_size, layer->stride, 
                        layer->padding, layer->dilation, layer->ceil_mode) + "\n";
                break;
            }
            case AdaptiveAvgPool2d_Backward: {
                AdaptiveAvgPool2d *layer = dynamic_cast<AdaptiveAvgPool2d *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);
                definition_file_gen += avgAdaptivePool2d_backward_CodeGen(kernel.parent_layer->layer_id, 
                        kernel.parent_layer->N, kernel.parent_layer->C, 
                        kernel.parent_layer->H, kernel.parent_layer->W, 
                        layer->outputsize_0, layer->outputsize_1) + "\n";
                break;
            }
            case Linear_Backward_Weight: {
                Linear *layer = dynamic_cast<Linear *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);
                definition_file_gen += linear_backward_weight_CodeGen(kernel.parent_layer->layer_id, 
                        kernel.parent_layer->N, kernel.parent_layer->C, 
                        kernel.parent_layer->H, kernel.parent_layer->W, 
                        layer->in_features, layer->out_features) + "\n";
                break;
            }
            case Linear_Backward_Input: {
                Linear *layer = dynamic_cast<Linear *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);
                definition_file_gen += linear_backward_input_CodeGen(kernel.parent_layer->layer_id, 
                        kernel.parent_layer->N, kernel.parent_layer->C, 
                        kernel.parent_layer->H, kernel.parent_layer->W, 
                        layer->in_features, layer->out_features) + "\n";
                break;
            }
            case Linear_Backward_Bias: {
                Linear *layer = dynamic_cast<Linear *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);
                definition_file_gen += linear_backward_bias_CodeGen(kernel.parent_layer->layer_id, 
                        kernel.parent_layer->N, kernel.parent_layer->C, 
                        kernel.parent_layer->H, kernel.parent_layer->W, 
                        layer->in_features, layer->out_features) + "\n";
                break;
            }
            case BatchNorm2d_Backward: {
                BatchNorm2d *layer = dynamic_cast<BatchNorm2d *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);
                definition_file_gen += batchnorm2d_backward_CodeGen(kernel.parent_layer->layer_id, 
                        kernel.parent_layer->N, kernel.parent_layer->C, 
                        kernel.parent_layer->H, kernel.parent_layer->W) + "\n";
                break;
            }
            case Add_MultiGredient: {
                Assert(kernel.parent_layer->other_d_outputs.size() > 0);
                definition_file_gen += multiGradient_add_CodeGen(kernel.parent_layer->layer_id, 
                        kernel.parent_layer->other_d_outputs.size() + 1, 
                        kernel.parent_layer->output_activation->size_in_byte / sizeof(float)) + "\n";
                break;
            }
            case Concat_Backward: {
                ConcatOp *layer = dynamic_cast<ConcatOp *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);
                definition_file_gen += concat_backward_CodeGen(kernel.parent_layer->layer_id, 
                        kernel.parent_layer->N, kernel.parent_layer->input_Cs, 
                        kernel.parent_layer->H, kernel.parent_layer->W) + "\n";
                break;
            }
            case Scale_Backward: {
                ScaleOp *layer = dynamic_cast<ScaleOp *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);
                definition_file_gen += scale_backward_CodeGen(kernel.parent_layer->layer_id, 
                        kernel.parent_layer->N, kernel.parent_layer->C, 
                        kernel.parent_layer->scale_H, kernel.parent_layer->scale_W) + "\n";
                break;
            }
            // Functions defined in utils
            case LoadData_A0: {
              A0_size = kernel.parent_layer->N * kernel.parent_layer->C * kernel.parent_layer->W * kernel.parent_layer->H;
            }
            case Dropout_Forward:
            case Conv2d_Apply_Grad:
            case BatchNorm2d_Apply_Grad:
            case Linear_Apply_Grad_Weight:
            case Linear_Apply_Grad_Bias:
            case Dropout_Backward:
            case makeLoss:
                break;
            default:
                Assert(false);
        }
        kernel_in_file_idx += need_specific_declaration(kernel.type) ? 1 : 0;
    }
    Assert(definition_file_gen.size() == 0);
    Assert(A0_size >= 0);

    // main function
    main_begin_gen += "// main function\n";
    main_begin_gen += "int main(int argc, const  char **argv) {\n";
    if (!is_individual) {
        if (is_UVM) {
            main_malloc_gen += "  // Unified malloc, cudaMallocManaged in final version\n";
            main_malloc_gen += "  CUDA_CHECK_RET( cudaMallocManaged(&dataset_base, num_iteration * " + 
                    to_string(A0_size * sizeof(float)) + ") );\n";
            main_malloc_gen += "  CUDA_CHECK_RET( cudaMallocManaged(&weight_base, weight_size_in_byte) );\n";
            main_malloc_gen += "  CUDA_CHECK_RET( cudaMallocManaged(&intermediate_base, intermediate_size_in_byte) );\n";
            main_malloc_gen += "\n";
        } else {
            main_malloc_gen += "  // Unified malloc, cudaMallocManaged in final version\n";
            main_malloc_gen += "  CUDA_CHECK_RET( cudaMalloc(&dataset_base, num_iteration * " + 
                    to_string(A0_size * sizeof(float)) + ") );\n";
            main_malloc_gen += "  CUDA_CHECK_RET( cudaMalloc(&weight_base, weight_size_in_byte) );\n";
            main_malloc_gen += "  CUDA_CHECK_RET( cudaMalloc(&intermediate_base, intermediate_size_in_byte) );\n";
            main_malloc_gen += "\n";
        }
    }

    // weight assignment
    if (!is_individual) {
        main_addr_assign_gen += "  // Weight addr assignment\n";
        for (int tensor_id = 0; tensor_id < tensor_list.size(); tensor_id++) {
            Tensor *tensor = tensor_list[tensor_id];
            const int num_prefilling_zeros = (int) ceil(log10(tensor_list.size())) - to_string(tensor_id).size();
            if (tensor->is_global_weight) {
                main_addr_assign_gen += "  " + tensor->name() + string(num_prefilling_zeros, ' ') + 
                        " = weight_base + " + 
                        to_string(tensor->address_offset / sizeof(float)) + ";\n";
            } else {
                main_addr_assign_gen += "  " + tensor->name() + string(num_prefilling_zeros, ' ') + 
                        " = intermediate_base + " + 
                        to_string(tensor->address_offset / sizeof(float)) + ";\n";
            }
        }
        main_addr_assign_gen += "\n";
    }

    // data initialization
    main_data_gen_gen += "  // Data & mask generation\n";
    for (CUDAKernel kernel : kernel_list) {
        switch (kernel.type) {
            case Conv2d_Forward: {
                Conv2d *layer = dynamic_cast<Conv2d *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);
                Tensor* weight_tensor = kernel.parent_layer->weight;
                Assert(weight_tensor->size_in_byte % sizeof(float) == 0);
                Assert(weight_tensor->is_global_weight);
                main_data_gen_gen += "  // Conv2d_Forward_l" + to_string(kernel.parent_layer->layer_id) + "\n";
                main_data_gen_gen += "  generate_data(" + weight_tensor->name() + ", " + 
                        to_string(weight_tensor->size_in_byte / sizeof(float)) + ");\n";
                break;
            }
            case Linear_Forward: {
                Linear *layer = dynamic_cast<Linear *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);
                Tensor* weight_tensor = kernel.parent_layer->weight;
                Tensor* bias_tensor = kernel.parent_layer->bias;
                Assert(weight_tensor->size_in_byte % sizeof(float) == 0);
                Assert(bias_tensor->size_in_byte % sizeof(float) == 0);
                Assert(weight_tensor->is_global_weight);
                Assert(bias_tensor->is_global_weight);
                main_data_gen_gen += "  // Linear_Forward_l" + to_string(kernel.parent_layer->layer_id) + "\n";
                main_data_gen_gen += "  generate_data(" + weight_tensor->name() + ", " + 
                        to_string(weight_tensor->size_in_byte / sizeof(float)) + ");\n";
                main_data_gen_gen += "  generate_data(" + bias_tensor->name() + ", " +
                        to_string(bias_tensor->size_in_byte / sizeof(float)) + ");\n";
                break;
            }
            case Dropout_Forward: {
                Dropout *layer = dynamic_cast<Dropout *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);
                long long size_in_byte = kernel.parent_layer->musk_array->size_in_byte;
                Assert(size_in_byte % sizeof(float) == 0);
                Assert(size_in_byte / sizeof(float) == kernel.parent_layer->N * kernel.parent_layer->C * 
                                                       kernel.parent_layer->H * kernel.parent_layer->W);
                main_data_gen_gen += "  // Dropout_Forward_l" + to_string(kernel.parent_layer->layer_id) + "\n";
                main_data_gen_gen += "  generate_mask(" + kernel.parent_layer->musk_array->name() + ", " + 
                        to_string(size_in_byte / sizeof(float)) + ", " + to_string(layer->p) + ");\n";
                break;
            }
            case BatchNorm2d_Forward: {
                BatchNorm2d *layer = dynamic_cast<BatchNorm2d *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);
                Tensor* gamma_beta_tensor = kernel.parent_layer->alpha_and_beta;
                Tensor* running_m_tensor = kernel.parent_layer->running_m;
                Tensor* running_v_tensor = kernel.parent_layer->running_v;
                Assert(gamma_beta_tensor->size_in_byte % sizeof(float) == 0);
                Assert(running_m_tensor->size_in_byte % sizeof(float) == 0);
                Assert(running_v_tensor->size_in_byte % sizeof(float) == 0);
                Assert(gamma_beta_tensor->is_global_weight);
                Assert(running_m_tensor->is_global_weight);
                Assert(running_v_tensor->is_global_weight);
                main_data_gen_gen += "  // BatchNorm2d_Forward_l" + to_string(kernel.parent_layer->layer_id) + "\n";
                main_data_gen_gen += "  generate_data(" + gamma_beta_tensor->name() + ", " + 
                        to_string(gamma_beta_tensor->size_in_byte / sizeof(float)) + ");\n";
                main_data_gen_gen += "  generate_data(" + running_m_tensor->name() + ", " + 
                        to_string(running_m_tensor->size_in_byte / sizeof(float)) + ");\n";
                main_data_gen_gen += "  generate_data(" + running_v_tensor->name() + ", " + 
                        to_string(running_v_tensor->size_in_byte / sizeof(float)) + ");\n";
                break;
            }
            case ReLU_Forward:
            case MaxPool2d_Forward:
            case AdaptiveAvgPool2d_Forward:
            default:
                break;
        }
    }
    main_data_gen_gen += "\n";

    // run main iteration, one iter for profiling and several iters for other simulations
    // TODO: refine these logic
    main_end_gen += "  // Run iteration\n";
    if (is_individual) {
        main_end_gen += "  run_iteration();\n";
    } else {
        main_end_gen += "  for (int i = 0; i < num_iteration; i++) {\n";
        main_end_gen += "    run_iteration(i);\n";
        main_end_gen += "  }\n";
    }
    main_end_gen += "\n";

    // data free
    main_end_gen += "  // Free\n";
    if (is_individual) {
        main_end_gen += "  cudaFree(workspace_base);\n";
    } else {
        main_end_gen += "  cudaFree(dataset_base);\n";
        main_end_gen += "  cudaFree(weight_base);\n";
        main_end_gen += "  cudaFree(intermediate_base);\n";
    }
    main_end_gen += "}\n";

    // run iteration
    // TODO: refine these logic
    string run_iter_prefix, run_iter_postfix1, run_iter_postfix2;
    if (is_individual) {
        run_iter_prefix   += "void run_iteration() {\n";
        run_iter_prefix   += "  cudaEvent_t start, stop;\n";
        run_iter_prefix   += "  float elapsedTime;\n";
        run_iter_prefix   += "\n";
        run_iter_prefix   += "  cudaEventCreate(&start);\n";
        run_iter_prefix   += "  cudaEventCreate(&stop);\n";
        run_iter_prefix   += "  cudaEventRecord(start, 0);\n";

        run_iter_postfix1 += "  cudaEventRecord(stop, 0);\n";
        run_iter_postfix1 += "  CUDA_CHECK_RET( cudaEventSynchronize(stop) );\n";
        run_iter_postfix1 += "  cudaEventElapsedTime(&elapsedTime, start, stop);\n";
        run_iter_postfix1 += "  printf(\"";
        run_iter_postfix2 += " %20.10f ms\\n\", elapsedTime);\n";
        run_iter_postfix2 += "}\n";
        run_iter_postfix2 += "\n";
    } else {
        run_iter_prefix   += "void run_iteration(int iter) {\n";
        run_iter_prefix   += "  cudaEvent_t start, stop;\n";
        run_iter_prefix   += "  float elapsedTime;\n";
        run_iter_prefix   += "\n";
        run_iter_prefix   += "  cudaEventCreate(&start);\n";
        run_iter_prefix   += "  cudaEventCreate(&stop);\n";
        run_iter_prefix   += "\n";

        run_iter_postfix1 += "}\n";
        run_iter_postfix1 += "\n";
    }

    int batch_size = forward_layers[0]->N;
    vector<string> dims;
    // loop through all kernels to produce main running loop
    for (CUDAKernel kernel : kernel_list) {
        if (!is_individual) {
            run_iter_gen += "  cudaEventRecord(start, 0);\n";
        }
        switch (kernel.type) {
            case Conv2d_Forward: {
                Conv2d *layer = dynamic_cast<Conv2d *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);

                string func_str = conv2d_forward_CodeGen(kernel.parent_layer->layer_id, 
                        kernel.parent_layer->H, kernel.parent_layer->W, 
                        kernel.parent_layer->N, kernel.parent_layer->C, 
                        layer->out_channels, layer->kernel_size_r, layer->kernel_size_s, 
                        layer->stride_0, layer->stride_1, layer->padding_0, layer->padding_1, layer->bias);
                Assert(get_raw_dims(func_str, dims) == 3);
                run_iter_gen += "  // In:" + dims[0] + ", Out:" + dims[1] + ", Weight:" + dims[2] + "\n";

                Tensor *input_tensor = kernel.parent_layer->input_activation;
                Tensor *output_tensor = kernel.parent_layer->output_activation;
                Tensor *weight_tensor = kernel.parent_layer->weight;
                Assert(get_dims(func_str, dims) == 3);
                run_iter_gen += "  forwardPass_conv2d_l" + to_string(kernel.parent_layer->layer_id) + 
                        "<<<" + to_string(batch_size) + ", " + to_string(num_threads) + ">>>(" +
                        "(float (*)" + dims[0] + ") " + input_tensor->name() + 
                        ", (float (*)" + dims[1] + ") " + output_tensor->name() + 
                        ", (float (*)" + dims[2] + ") " + weight_tensor->name() + 
                        ");\n";
                break;
            }
            case ReLU_Forward: {
                ReLU *layer = dynamic_cast<ReLU *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);

                string func_str = reLU_forward_CodeGen(kernel.parent_layer->layer_id, 
                        kernel.parent_layer->N, kernel.parent_layer->C, 
                        kernel.parent_layer->H, kernel.parent_layer->W);
                Assert(get_raw_dims(func_str, dims) == 2);
                run_iter_gen += "  // In:" + dims[0] + ", Out:" + dims[1] + "\n";

                Tensor *input_tensor = kernel.parent_layer->input_activation;
                Tensor *output_tensor = kernel.parent_layer->output_activation;
                Assert(get_dims(func_str, dims) == 2);
                run_iter_gen += "  forwardPass_ReLU_l" + to_string(kernel.parent_layer->layer_id) + 
                        "<<<" + to_string(batch_size) + ", " + to_string(num_threads) + ">>>(" +
                        "(float (*)" + dims[0] + ") " + input_tensor->name() + 
                        ", (float (*)" + dims[1] + ") " + output_tensor->name() + 
                        ");\n";
                break;
            }
            case MaxPool2d_Forward: {
                MaxPool2d *layer = dynamic_cast<MaxPool2d *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);

                string func_str = maxPool2d_forward_CodeGen(kernel.parent_layer->layer_id, 
                        kernel.parent_layer->N, kernel.parent_layer->C, 
                        kernel.parent_layer->H, kernel.parent_layer->W, 
                        layer->kernel_size, layer->kernel_size, layer->stride, 
                        layer->padding, layer->dilation, layer->ceil_mode);
                Assert(get_raw_dims(func_str, dims) == 2);
                run_iter_gen += "  // In:" + dims[0] + ", Out:" + dims[1] + "\n";

                Tensor *input_tensor = kernel.parent_layer->input_activation;
                Tensor *output_tensor = kernel.parent_layer->output_activation;
                Assert(get_dims(func_str, dims) == 2);
                run_iter_gen += "  forwardPass_MaxPool2d_l" + to_string(kernel.parent_layer->layer_id) + 
                        "<<<" + to_string(batch_size) + ", " + to_string(num_threads) + ">>>(" +
                        "(float (*)" + dims[0] + ") " + input_tensor->name() + 
                        ", (float (*)" + dims[1] + ") " + output_tensor->name() + 
                        ");\n";
                break;
            }
            case AdaptiveAvgPool2d_Forward: {
                AdaptiveAvgPool2d *layer = dynamic_cast<AdaptiveAvgPool2d *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);

                string func_str = avgAdaptivePool2d_forward_CodeGen(kernel.parent_layer->layer_id, 
                        kernel.parent_layer->N, kernel.parent_layer->C, 
                        kernel.parent_layer->H, kernel.parent_layer->W, 
                        layer->outputsize_0, layer->outputsize_1);
                Assert(get_raw_dims(func_str, dims) == 2);
                run_iter_gen += "  // In:" + dims[0] + ", Out:" + dims[1] + "\n";

                Tensor *input_tensor = kernel.parent_layer->input_activation;
                Tensor *output_tensor = kernel.parent_layer->output_activation;
                Assert(get_dims(func_str, dims) == 2);
                run_iter_gen += "  forwardPass_avgAdaptivePool2d_l" + to_string(kernel.parent_layer->layer_id) + 
                        "<<<" + to_string(batch_size) + ", " + to_string(num_threads) + ">>>(" +
                        "(float (*)" + dims[0] + ") " + input_tensor->name() + 
                        ", (float (*)" + dims[1] + ") " + output_tensor->name() + 
                        ");\n";
                break;
            }
            case Linear_Forward: {
                Linear *layer = dynamic_cast<Linear *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);

                string func_str = linear_forward_CodeGen(kernel.parent_layer->layer_id, 
                        kernel.parent_layer->N, kernel.parent_layer->C, 
                        kernel.parent_layer->H, kernel.parent_layer->W, 
                        layer->in_features, layer->out_features);
                Assert(get_raw_dims(func_str, dims) == 4);
                run_iter_gen += "  // In:" + dims[0] + ", Out:" + dims[1] + 
                        ", Weight:" + dims[2] + ", Bias:" + dims[3] + "\n";

                Tensor *input_tensor = kernel.parent_layer->input_activation;
                Tensor *output_tensor = kernel.parent_layer->output_activation;
                Tensor *weight_tensor = kernel.parent_layer->weight;
                Tensor *bias_tensor = kernel.parent_layer->bias;
                Assert(get_dims(func_str, dims) == 4);
                run_iter_gen += "  forwardPass_Linear_l" + to_string(kernel.parent_layer->layer_id) + 
                        "<<<" + to_string(batch_size) + ", " + to_string(num_threads) + ">>>(" +
                        "(float (*)" + dims[0] + ") " + input_tensor->name() + 
                        ", (float (*)" + dims[1] + ") " + output_tensor->name() + 
                        ", (float (*)" + dims[2] + ") " + weight_tensor->name() + 
                        ", (float (*)" + dims[3] + ") " + bias_tensor->name() + 
                        ");\n";
                break;
            }
            case Dropout_Forward: {
                Dropout *layer = dynamic_cast<Dropout *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);

                long long size_in_byte = kernel.parent_layer->musk_array->size_in_byte;
                Assert(size_in_byte % sizeof(float) == 0);
                Assert(size_in_byte / sizeof(float) == kernel.parent_layer->N * kernel.parent_layer->C * 
                                                       kernel.parent_layer->H * kernel.parent_layer->W);
                run_iter_gen += "  // Size: " + to_string(size_in_byte / sizeof(float)) + "\n";

                Tensor *input_tensor = kernel.parent_layer->input_activation;
                Tensor *output_tensor = kernel.parent_layer->output_activation;
                Tensor *dropout_tensor = kernel.parent_layer->musk_array;
                run_iter_gen += "  forwardPass_Dropout<<<" + to_string(batch_size) + ", " + to_string(num_threads) + ">>>(" + 
                        input_tensor->name() + ", " + 
                        output_tensor->name() + ", " + 
                        dropout_tensor->name() + ", " + 
                        to_string(size_in_byte / sizeof(float)) + ");\n";
                break;
            }
            case BatchNorm2d_Forward: {
                BatchNorm2d *layer = dynamic_cast<BatchNorm2d *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);

                string func_str = batchnorm2d_forward_CodeGen(kernel.parent_layer->layer_id, 
                        kernel.parent_layer->N, kernel.parent_layer->C, 
                        kernel.parent_layer->H, kernel.parent_layer->W, 
                        layer->eps, layer->momentum, layer->track_running_stats);
                Assert(get_raw_dims(func_str, dims) == 9);
                run_iter_gen += "  // In:" + dims[0] + ", Gamma_beta:" + dims[1] + ", Running_m:" + dims[2] + 
                        ", Running_v:" + dims[3] + ", Mu:" + dims[4] + ", Var:" + dims[5] + 
                        ", V1:" + dims[6] + ", V2:" + dims[7] + ", Output:" + dims[8] + "\n";

                Tensor *input_tensor = kernel.parent_layer->input_activation;
                Tensor *gamma_beta_tensor = kernel.parent_layer->alpha_and_beta;
                Tensor *running_m_tensor = kernel.parent_layer->running_m;
                Tensor *running_v_tensor = kernel.parent_layer->running_v;
                Tensor *mu_tensor = kernel.parent_layer->mu;
                Tensor *var_tensor = kernel.parent_layer->var;
                Tensor *v1_tensor = kernel.parent_layer->v1;
                Tensor *v2_tensor = kernel.parent_layer->v2;
                Tensor *output_tensor = kernel.parent_layer->output_activation;
                Assert(get_dims(func_str, dims) == 9);
                run_iter_gen += "  forwardPass_BatchNorm2d_l" + to_string(kernel.parent_layer->layer_id) + 
                        "<<<" + to_string(batch_size) + ", " + to_string(num_threads) + ">>>(" +
                        "(float (*)" + dims[0] + ") " + input_tensor->name() + 
                        ", (float (*)" + dims[1] + ") " + gamma_beta_tensor->name() + 
                        ", (float (*)" + dims[2] + ") " + running_m_tensor->name() + 
                        ", (float (*)" + dims[3] + ") " + running_v_tensor->name() + 
                        ", (float (*)" + dims[4] + ") " + mu_tensor->name() + 
                        ", (float (*)" + dims[5] + ") " + var_tensor->name() + 
                        ", (float (*)" + dims[6] + ") " + v1_tensor->name() + 
                        ", (float (*)" + dims[7] + ") " + v2_tensor->name() + 
                        ", (float (*)" + dims[8] + ") " + output_tensor->name() + 
                        ");\n";
                break;
            }
            case Add_Forward: {
                AddOp *layer = dynamic_cast<AddOp *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);

                Tensor *input_tensor = kernel.parent_layer->input_activation;
                Tensor *output_tensor = kernel.parent_layer->output_activation;
                long long size_in_byte = kernel.parent_layer->output_activation->size_in_byte;
                Assert(size_in_byte % sizeof(float) == 0);

                run_iter_gen += "  // NumInputs:" + to_string(kernel.parent_layer->other_inputs.size() + 1) + 
                        ", Size:" + to_string(size_in_byte / sizeof(float)) + "\n";

                run_iter_gen += "  forwardPass_Add_l" + to_string(kernel.parent_layer->layer_id) + 
                        "<<<" + to_string(batch_size) + ", " + to_string(num_threads) + ">>>(" + 
                        input_tensor->name() + ", ";
                for (int i = 0; i < kernel.parent_layer->other_inputs.size(); i++) {
                    Tensor *other_input_tensor = kernel.parent_layer->other_inputs[i];
                    Assert(size_in_byte == other_input_tensor->size_in_byte);
                    run_iter_gen += other_input_tensor->name() + ", ";
                }
                run_iter_gen += output_tensor->name() + ", " + to_string(size_in_byte / sizeof(float)) + ");\n";
                break;
            }
            case Concat_Forward: {
                ConcatOp *layer = dynamic_cast<ConcatOp *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);

                int out_channel = 0;
                string func_str = concat_forward_CodeGen(kernel.parent_layer->layer_id, 
                        kernel.parent_layer->N, kernel.parent_layer->input_Cs, 
                        kernel.parent_layer->H, kernel.parent_layer->W);
                Assert(kernel.parent_layer->input_Cs.size() >= 1);
                Assert(get_raw_dims(func_str, dims) == kernel.parent_layer->input_Cs.size() + 1);
                run_iter_gen += "  // ";
                for (int in_index = 0; in_index < kernel.parent_layer->input_Cs.size(); in_index++) {
                    int in_channel = kernel.parent_layer->input_Cs[in_index];
                    out_channel += in_channel;
                    run_iter_gen += "In" + to_string(in_index) + ": " + dims[in_index] + ", ";
                }
                run_iter_gen += "Out: " + dims[kernel.parent_layer->input_Cs.size()] + "\n";

                Tensor *input_tensor = kernel.parent_layer->input_activation;
                Tensor *output_tensor = kernel.parent_layer->output_activation;
                Assert(get_dims(func_str, dims) == kernel.parent_layer->input_Cs.size() + 1);
                run_iter_gen += "  concat_forward_l" + to_string(kernel.parent_layer->layer_id) + 
                        "<<<" + to_string(batch_size) + ", " + to_string(num_threads) + ">>>(" +
                        "(float (*)" + dims[0] + ") " + input_tensor->name() + ", ";
                for (int i = 1; i < kernel.parent_layer->input_Cs.size(); i++) {
                    run_iter_gen += "(float (*)" + dims[i] + ") " + 
                            kernel.parent_layer->other_inputs[i - 1]->name() + ", ";
                }
                run_iter_gen += "(float (*)" + dims[kernel.parent_layer->input_Cs.size()] + ") " + 
                        output_tensor->name() + ");\n";
                break;
            }
            case Scale_Forward: {
                ScaleOp *layer = dynamic_cast<ScaleOp *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);

                string func_str = scale_forward_CodeGen(kernel.parent_layer->layer_id, 
                        kernel.parent_layer->N, kernel.parent_layer->C, 
                        kernel.parent_layer->scale_H, kernel.parent_layer->scale_W);
                Assert(get_raw_dims(func_str, dims) == 2);
                run_iter_gen += "  // In:" + dims[0] + ", Out:" + dims[1] + "\n";

                Tensor *input_tensor = kernel.parent_layer->input_activation;
                Tensor *output_tensor = kernel.parent_layer->output_activation;
                Assert(get_dims(func_str, dims) == 2);
                run_iter_gen += "  scale_forward_l" + to_string(kernel.parent_layer->layer_id) + 
                        "<<<" + to_string(batch_size) + ", " + to_string(num_threads) + ">>>(" +
                        "(float (*)" + dims[0] + ") " + input_tensor->name() + 
                        ", (float (*)" + dims[1] + ") " + output_tensor->name() + 
                        ");\n";
                break;
            }
            case Conv2d_Backward_Weight: {
                Conv2d *layer = dynamic_cast<Conv2d *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);

                string func_str = conv2d_backward_weight_CodeGen(kernel.parent_layer->layer_id, 
                        kernel.parent_layer->H, kernel.parent_layer->W, 
                        kernel.parent_layer->N, kernel.parent_layer->C, 
                        layer->out_channels, layer->kernel_size_r, layer->kernel_size_s, 
                        layer->stride_0, layer->stride_1, layer->padding_0, layer->padding_1, layer->bias);
                Assert(get_raw_dims(func_str, dims) == 3);
                run_iter_gen += "  // In:" + dims[0] + ", D_out:" + dims[1] + ", D_weight:" + dims[2] + "\n";

                Tensor *input_tensor = kernel.parent_layer->input_activation;
                Tensor *d_output_tensor = kernel.parent_layer->d_output;
                Tensor *d_weight_tensor = kernel.parent_layer->d_weight;
                Assert(get_dims(func_str, dims) == 3);
                run_iter_gen += "  backwardPass_conv2d_Weight_l" + to_string(kernel.parent_layer->layer_id) + 
                        "<<<" + to_string(batch_size) + ", " + to_string(num_threads) + ">>>(" +
                        "(float (*)" + dims[0] + ") " + input_tensor->name() + 
                        ", (float (*)" + dims[1] + ") " + d_output_tensor->name() + 
                        ", (float (*)" + dims[2] + ") " + d_weight_tensor->name() + 
                        ");\n";
                break;
            }
            case Conv2d_Backward_Input: {
                Conv2d *layer = dynamic_cast<Conv2d *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);

                string func_str = conv2d_backward_input_CodeGen(kernel.parent_layer->layer_id, 
                        kernel.parent_layer->H, kernel.parent_layer->W, 
                        kernel.parent_layer->N, kernel.parent_layer->C, 
                        layer->out_channels, layer->kernel_size_r, layer->kernel_size_s, 
                        layer->stride_0, layer->stride_1, layer->padding_0, layer->padding_1, layer->bias);
                Assert(get_raw_dims(func_str, dims) == 3);
                run_iter_gen += "  // D_in:" + dims[0] + ", D_out:" + dims[1] + ", Weight:" + dims[2] + "\n";

                Tensor *d_input_tensor = kernel.parent_layer->d_input;
                Tensor *d_output_tensor = kernel.parent_layer->d_output;
                Tensor *weight_tensor = kernel.parent_layer->weight;
                Assert(get_dims(func_str, dims) == 3);
                run_iter_gen += "  backwardPass_conv2d_Input_l" + to_string(kernel.parent_layer->layer_id) + 
                        "<<<" + to_string(batch_size) + ", " + to_string(num_threads) + ">>>(" +
                        "(float (*)" + dims[0] + ") " + d_input_tensor->name() + 
                        ", (float (*)" + dims[1] + ") " + d_output_tensor->name() + 
                        ", (float (*)" + dims[2] + ") " + weight_tensor->name() + 
                        ");\n";
                break;
            }
            case Conv2d_Apply_Grad: {
                Conv2d *layer = dynamic_cast<Conv2d *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);

                long long size_in_byte = kernel.parent_layer->weight->size_in_byte;
                Assert(size_in_byte % sizeof(float) == 0);
                run_iter_gen += "  // Conv2d Size: " + to_string(size_in_byte / sizeof(float)) + "\n";

                Tensor *weight_tensor = kernel.parent_layer->weight;
                Tensor *d_weight_tensor = kernel.parent_layer->d_weight;
                Assert(size_in_byte == kernel.parent_layer->d_weight->size_in_byte);
                run_iter_gen += "  apply_grad<<<" + to_string(batch_size) + ", " + to_string(num_threads) + ">>>(" + 
                        weight_tensor->name() + ", " + 
                        d_weight_tensor->name() + ", " + 
                        to_string(size_in_byte / sizeof(float)) + ");\n";
                break;
            }
            case ReLU_Backward: {
                ReLU *layer = dynamic_cast<ReLU *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);

                string func_str = reLU_backward_CodeGen(kernel.parent_layer->layer_id, 
                        kernel.parent_layer->N, kernel.parent_layer->C, 
                        kernel.parent_layer->H, kernel.parent_layer->W);
                Assert(get_raw_dims(func_str, dims) == 3);
                run_iter_gen += "  // D_in:" + dims[0] + ", In:" + dims[1] + ", D_output:" + dims[2] + "\n";

                Tensor *d_input_tensor = kernel.parent_layer->d_input;
                Tensor *input_tensor = kernel.parent_layer->input_activation;
                Tensor *d_output_tensor = kernel.parent_layer->d_output;
                Assert(get_dims(func_str, dims) == 3);
                run_iter_gen += "  backwardPass_ReLU_l" + to_string(kernel.parent_layer->layer_id) + 
                        "<<<" + to_string(batch_size) + ", " + to_string(num_threads) + ">>>(" +
                        "(float (*)" + dims[0] + ") " + d_input_tensor->name() + 
                        ", (float (*)" + dims[1] + ") " + input_tensor->name() + 
                        ", (float (*)" + dims[1] + ") " + d_output_tensor->name() + 
                        ");\n";
                break;
            }
            case MaxPool2d_Backward: {
                MaxPool2d *layer = dynamic_cast<MaxPool2d *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);

                string func_str = maxPool2d_backward_CodeGen(kernel.parent_layer->layer_id, 
                        kernel.parent_layer->N, kernel.parent_layer->C, 
                        kernel.parent_layer->H, kernel.parent_layer->W, 
                        layer->kernel_size, layer->kernel_size, layer->stride, 
                        layer->padding, layer->dilation, layer->ceil_mode);
                Assert(get_raw_dims(func_str, dims) == 3);
                run_iter_gen += "  // D_in:" + dims[0] + ", In:" + dims[1] + ", D_output:" + dims[2] + "\n";

                Tensor *d_input_tensor = kernel.parent_layer->d_input;
                Tensor *input_tensor = kernel.parent_layer->input_activation;
                Tensor *d_output_tensor = kernel.parent_layer->d_output;
                Assert(get_dims(func_str, dims) == 3);
                run_iter_gen += "  backwardPass_MaxPool2d_l" + to_string(kernel.parent_layer->layer_id) + 
                        "<<<" + to_string(batch_size) + ", " + to_string(num_threads) + ">>>(" +
                        "(float (*)" + dims[0] + ") " + d_input_tensor->name() + 
                        ", (float (*)" + dims[1] + ") " + input_tensor->name() + 
                        ", (float (*)" + dims[2] + ") " + d_output_tensor->name() + 
                        ");\n";
                break;
            }
            case AdaptiveAvgPool2d_Backward: {
                AdaptiveAvgPool2d *layer = dynamic_cast<AdaptiveAvgPool2d *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);

                string func_str = avgAdaptivePool2d_backward_CodeGen(kernel.parent_layer->layer_id, 
                        kernel.parent_layer->N, kernel.parent_layer->C, 
                        kernel.parent_layer->H, kernel.parent_layer->W, 
                        layer->outputsize_0, layer->outputsize_1);
                Assert(get_raw_dims(func_str, dims) == 2);
                run_iter_gen += "  // D_in:" + dims[0] + ", D_out:" + dims[1] + "\n";

                Tensor *d_input_tensor = kernel.parent_layer->d_input;
                Tensor *d_output_tensor = kernel.parent_layer->d_output;
                Assert(get_dims(func_str, dims) == 2);
                run_iter_gen += "  backwardPass_avgAdaptivePool2d_l" + to_string(kernel.parent_layer->layer_id) + 
                        "<<<" + to_string(batch_size) + ", " + to_string(num_threads) + ">>>(" +
                        "(float (*)" + dims[0] + ") " + d_input_tensor->name() + 
                        ", (float (*)" + dims[1] + ") " + d_output_tensor->name() + 
                        ");\n";
                break;
            }
            case Linear_Backward_Weight: {
                Linear *layer = dynamic_cast<Linear *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);

                string func_str = linear_backward_weight_CodeGen(kernel.parent_layer->layer_id, 
                        kernel.parent_layer->N, kernel.parent_layer->C, 
                        kernel.parent_layer->H, kernel.parent_layer->W, 
                        layer->in_features, layer->out_features);
                Assert(get_raw_dims(func_str, dims) == 3);
                run_iter_gen += "  // In:" + dims[0] + ", D_out:" + dims[1] + ", D_weight:" + dims[2] + "\n";

                Tensor *input_tensor = kernel.parent_layer->input_activation;
                Tensor *d_output_tensor = kernel.parent_layer->d_output;
                Tensor *d_weight_tensor = kernel.parent_layer->d_weight;
                Assert(get_dims(func_str, dims) == 3);
                run_iter_gen += "  backwardPass_Linear_weight_l" + to_string(kernel.parent_layer->layer_id) + 
                        "<<<" + to_string(batch_size) + ", " + to_string(num_threads) + ">>>(" +
                        "(float (*)" + dims[0] + ") " + input_tensor->name() + 
                        ", (float (*)" + dims[1] + ") " + d_output_tensor->name() + 
                        ", (float (*)" + dims[2] + ") " + d_weight_tensor->name() + 
                        ");\n";
                break;
            }
            case Linear_Backward_Input: {
                Linear *layer = dynamic_cast<Linear *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);

                string func_str = linear_backward_input_CodeGen(kernel.parent_layer->layer_id, 
                        kernel.parent_layer->N, kernel.parent_layer->C, 
                        kernel.parent_layer->H, kernel.parent_layer->W, 
                        layer->in_features, layer->out_features);
                Assert(get_raw_dims(func_str, dims) == 3);
                run_iter_gen += "  // D_in:" + dims[0] + ", D_out:" + dims[1] + ", Weight:" + dims[2] + "\n";

                Tensor *d_input_tensor = kernel.parent_layer->d_input;
                Tensor *d_output_tensor = kernel.parent_layer->d_output;
                Tensor *weight_tensor = kernel.parent_layer->weight;
                Assert(get_dims(func_str, dims) == 3);
                run_iter_gen += "  backwardPass_Linear_input_l" + to_string(kernel.parent_layer->layer_id) + 
                        "<<<" + to_string(batch_size) + ", " + to_string(num_threads) + ">>>(" +
                        "(float (*)" + dims[0] + ") " + d_input_tensor->name() + 
                        ", (float (*)" + dims[1] + ") " + d_output_tensor->name() + 
                        ", (float (*)" + dims[2] + ") " + weight_tensor->name() + 
                        ");\n";
                break;
            }
            case Linear_Backward_Bias: {
                Linear *layer = dynamic_cast<Linear *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);

                string func_str = linear_backward_bias_CodeGen(kernel.parent_layer->layer_id, 
                        kernel.parent_layer->N, kernel.parent_layer->C, 
                        kernel.parent_layer->H, kernel.parent_layer->W, 
                        layer->in_features, layer->out_features);
                Assert(get_raw_dims(func_str, dims) == 2);
                run_iter_gen += "  // D_out:" + dims[0] + ", D_bias:" + dims[1] + "\n";

                Tensor *d_output_tensor = kernel.parent_layer->d_output;
                Tensor *d_bias_tensor = kernel.parent_layer->d_bias;
                Assert(get_dims(func_str, dims) == 2);
                run_iter_gen += "  backwardPass_Linear_bias_l" + to_string(kernel.parent_layer->layer_id) + 
                        "<<<" + to_string(batch_size) + ", " + to_string(num_threads) + ">>>(" +
                        "(float (*)" + dims[0] + ") " + d_output_tensor->name() + 
                        ", (float (*)" + dims[1] + ") " + d_bias_tensor->name() + 
                        ");\n";
                break;
            }
            case Linear_Apply_Grad_Bias: {
                Linear *layer = dynamic_cast<Linear *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);

                long long size_in_byte = kernel.parent_layer->bias->size_in_byte;
                Assert(size_in_byte % sizeof(float) == 0);
                run_iter_gen += "  // Linear Bias Size: " + to_string(size_in_byte / sizeof(float)) + "\n";

                Tensor *bias_tensor = kernel.parent_layer->bias;
                Tensor *d_bias_tensor = kernel.parent_layer->d_bias;
                Assert(size_in_byte == kernel.parent_layer->d_bias->size_in_byte);
                run_iter_gen += "  apply_grad<<<" + to_string(batch_size) + ", " + to_string(num_threads) + ">>>(" + 
                        bias_tensor->name() + ", " + 
                        d_bias_tensor->name() + ", " + 
                        to_string(size_in_byte / sizeof(float)) + ");\n";
                break;
            }
            case Linear_Apply_Grad_Weight: {
                Linear *layer = dynamic_cast<Linear *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);

                long long size_in_byte = kernel.parent_layer->weight->size_in_byte;
                Assert(size_in_byte % sizeof(float) == 0);
                run_iter_gen += "  // Linear Weight Size: " + to_string(size_in_byte / sizeof(float)) + "\n";

                Tensor *weight_tensor = kernel.parent_layer->weight;
                Tensor *d_weight_tensor = kernel.parent_layer->d_weight;
                Assert(size_in_byte == kernel.parent_layer->d_weight->size_in_byte);
                run_iter_gen += "  apply_grad<<<" + to_string(batch_size) + ", " + to_string(num_threads) + ">>>(" + 
                        weight_tensor->name() + ", " + 
                        d_weight_tensor->name() + ", " + 
                        to_string(size_in_byte / sizeof(float)) + ");\n";
                break;
            }
            case Dropout_Backward: {
                Dropout *layer = dynamic_cast<Dropout *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);

                long long size_in_byte = kernel.parent_layer->output_activation->size_in_byte;
                Assert(size_in_byte % sizeof(float) == 0);
                Assert(size_in_byte / sizeof(float) == kernel.parent_layer->N * kernel.parent_layer->C * 
                                                       kernel.parent_layer->H * kernel.parent_layer->W);
                run_iter_gen += "  // Size: " + to_string(size_in_byte / sizeof(float)) + "\n";

                Tensor *d_output_tensor = kernel.parent_layer->d_output;
                Tensor *d_input_tensor = kernel.parent_layer->d_input;
                Tensor *mask_tensor = kernel.parent_layer->musk_array;
                Assert(size_in_byte == kernel.parent_layer->d_output->size_in_byte);
                run_iter_gen +=
                        "  backwardPass_Dropout<<<" + to_string(batch_size) + ", " + to_string(num_threads) + ">>>(" +
                        d_output_tensor->name() + ", " +
                        d_input_tensor->name() + ", " +
                        mask_tensor->name() + ", " +
                        to_string(size_in_byte / sizeof(float)) + ");\n";
                break;
            }
            case BatchNorm2d_Backward: {
                BatchNorm2d *layer = dynamic_cast<BatchNorm2d *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);

                string func_str = batchnorm2d_backward_CodeGen(kernel.parent_layer->layer_id, 
                        kernel.parent_layer->N, kernel.parent_layer->C, 
                        kernel.parent_layer->H, kernel.parent_layer->W);
                Assert(get_raw_dims(func_str, dims) == 10);
                run_iter_gen += "  // D_out:" + dims[0] + ", V1:" + dims[1] + ", V2:" + dims[2] +
                        ", Gamma_beta:" + dims[3] + ", D_v1:" + dims[4] + ", D_v2:" + dims[5] +
                        ", D_var:" + dims[6] + ", D_mu:" + dims[7] + ", D_input:" + dims[8] +
                        ", D_gamma_beta:" + dims[9] + "\n";

                Tensor *d_output_tensor = kernel.parent_layer->d_output;
                Tensor *v1_tensor = kernel.parent_layer->v1;
                Tensor *v2_tensor = kernel.parent_layer->v2;
                Tensor *gamma_beta_tensor = kernel.parent_layer->alpha_and_beta;
                Tensor *d_v1_tensor = kernel.parent_layer->d_v1;
                Tensor *d_v2_tensor = kernel.parent_layer->d_v2;
                Tensor *d_var_tensor = kernel.parent_layer->d_var;
                Tensor *d_mu_tensor = kernel.parent_layer->d_mu;
                Tensor *d_input_tensor = kernel.parent_layer->d_input;
                Tensor *d_gamma_beta_tensor = kernel.parent_layer->d_alpha_and_beta;
                Assert(get_dims(func_str, dims) == 10);
                run_iter_gen += "  backwardPass_BatchNorm2d_l" + to_string(kernel.parent_layer->layer_id) +
                        "<<<" + to_string(batch_size) + ", " + to_string(num_threads) + ">>>(" +
                        "(float (*)" + dims[0] + ") " + d_output_tensor->name() +
                        ", (float (*)" + dims[1] + ") " + v1_tensor->name() +
                        ", (float (*)" + dims[2] + ") " + v2_tensor->name() +
                        ", (float (*)" + dims[3] + ") " + gamma_beta_tensor->name() +
                        ", (float (*)" + dims[4] + ") " + d_v1_tensor->name() +
                        ", (float (*)" + dims[5] + ") " + d_v2_tensor->name() +
                        ", (float (*)" + dims[6] + ") " + d_var_tensor->name() +
                        ", (float (*)" + dims[7] + ") " + d_mu_tensor->name() +
                        ", (float (*)" + dims[8] + ") " + d_input_tensor->name() +
                        ", (float (*)" + dims[9] + ") " + d_gamma_beta_tensor->name() +
                        ");\n";
                break;
            }
            case BatchNorm2d_Apply_Grad: {
                BatchNorm2d *layer = dynamic_cast<BatchNorm2d *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);

                long long size_in_byte = kernel.parent_layer->alpha_and_beta->size_in_byte;
                Assert(size_in_byte % sizeof(float) == 0);
                run_iter_gen += "  // BatchNorm Size: " + to_string(size_in_byte / sizeof(float)) + "\n";

                Tensor *gamma_beta_tensor = kernel.parent_layer->alpha_and_beta;
                Tensor *d_gamma_beta_tensor = kernel.parent_layer->d_alpha_and_beta;
                run_iter_gen +=
                        "  apply_grad<<<" + to_string(batch_size) + ", " + to_string(num_threads) + ">>>(" +
                        gamma_beta_tensor->name() + ", " +
                        d_gamma_beta_tensor->name() + ", " +
                        to_string(size_in_byte / sizeof(float)) + ");\n";
                break;
            }
            case LoadData_A0: {
                run_iter_gen += "  // TransferA0 Size: " + to_string(A0_size) + "\n";
                run_iter_gen +=
                        "  transfer_A0<<<" + to_string(batch_size) + ", " + to_string(num_threads) + ">>>(" +
                        "&dataset_base[iter * " + to_string(A0_size) + "], " +
                        kernel.parent_layer->input_activation->name() + ", " +
                        to_string(A0_size) + ");\n";
                break;
            }
            case makeLoss: {
                Tensor *error_tensor = kernel.parent_layer->d_output;
                Tensor *output_tensor = kernel.parent_layer->output_activation;
                const int Y = 0;
                long long size_in_byte = error_tensor->size_in_byte;
                Assert(size_in_byte == output_tensor->size_in_byte);
                run_iter_gen += "  // MakeError Size: " + to_string(size_in_byte) + "\n";
                run_iter_gen +=
                        "  makeError<<<" + to_string(batch_size) + ", " + to_string(num_threads) + ">>>(" +
                        error_tensor->name() + ", " +
                        output_tensor->name() + ", " +
                        to_string(Y) + ", " +
                        to_string(size_in_byte / sizeof(float)) + ");\n";
                break;
            }
            case Add_MultiGredient: {
                Tensor *d_output_tensor = kernel.parent_layer->d_output;
                long long size_in_byte = kernel.parent_layer->d_output->size_in_byte;
                Assert(size_in_byte % sizeof(float) == 0);

                run_iter_gen += "  // NumD_out:" + to_string(kernel.parent_layer->other_inputs.size() + 1) +
                        ", Size:" + to_string(size_in_byte / sizeof(float)) + "\n";

                run_iter_gen += "  multiGradient_Add_l" + to_string(kernel.parent_layer->layer_id) +
                        "<<<" + to_string(batch_size) + ", " + to_string(num_threads) + ">>>(" +
                        d_output_tensor->name() + ", ";
                for (int i = 0; i < kernel.parent_layer->other_d_outputs.size(); i++) {
                    Tensor *other_d_output_tensor = kernel.parent_layer->other_d_outputs[i];
                    Assert(size_in_byte == other_d_output_tensor->size_in_byte);
                    run_iter_gen += other_d_output_tensor->name() + ", ";
                }
                run_iter_gen += to_string(size_in_byte / sizeof(float)) + ");\n";
                break;
            }
            case Concat_Backward: {
                ConcatOp *layer = dynamic_cast<ConcatOp *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);

                int out_channel = 0;
                string func_str = concat_backward_CodeGen(kernel.parent_layer->layer_id,
                        kernel.parent_layer->N, kernel.parent_layer->input_Cs,
                        kernel.parent_layer->H, kernel.parent_layer->W);
                Assert(kernel.parent_layer->input_Cs.size() >= 1);
                Assert(get_raw_dims(func_str, dims) == kernel.parent_layer->input_Cs.size() + 1);
                run_iter_gen += "  // ";
                for (int in_index = 0; in_index < kernel.parent_layer->input_Cs.size(); in_index++) {
                    int in_channel = kernel.parent_layer->input_Cs[in_index];
                    out_channel += in_channel;
                    run_iter_gen += "D_in" + to_string(in_index) + ": " + dims[in_index] + ", ";
                }
                run_iter_gen += "D_out: " + dims[kernel.parent_layer->input_Cs.size()] + "\n";

                Tensor *d_input_tensor = kernel.parent_layer->d_input;
                Tensor *d_output_tensor = kernel.parent_layer->d_output;
                Assert(get_dims(func_str, dims) == kernel.parent_layer->input_Cs.size() + 1);
                run_iter_gen += "  concat_backward_l" + to_string(kernel.parent_layer->layer_id) +
                        "<<<" + to_string(batch_size) + ", " + to_string(num_threads) + ">>>(" +
                        "(float (*)" + dims[0] + ") " + d_input_tensor->name() + ", ";
                for (int i = 1; i < kernel.parent_layer->input_Cs.size(); i++) {
                    run_iter_gen += "(float (*)" + dims[i] + ") " +
                            kernel.parent_layer->other_d_inputs[i - 1]->name() + ", ";
                }
                run_iter_gen += "(float (*)" + dims[kernel.parent_layer->input_Cs.size()] + ") " +
                        d_output_tensor->name() + ");\n";
                break;
            }
            case Scale_Backward: {
                ScaleOp *layer = dynamic_cast<ScaleOp *>(kernel.parent_layer->operatorr);
                Assert(layer != nullptr);

                string func_str = scale_backward_CodeGen(kernel.parent_layer->layer_id,
                    kernel.parent_layer->N, kernel.parent_layer->C,
                    kernel.parent_layer->scale_H, kernel.parent_layer->scale_W);
                Assert(get_raw_dims(func_str, dims) == 2);
                run_iter_gen += "  // D_in:" + dims[0] + ", D_out:" + dims[1] + "\n";

                Tensor *d_input_tensor = kernel.parent_layer->d_input;
                Tensor *d_output_tensor = kernel.parent_layer->d_output;
                Assert(get_dims(func_str, dims) == 2);
                run_iter_gen += "  scale_backward_l" + to_string(kernel.parent_layer->layer_id) +
                        "<<<" + to_string(batch_size) + ", " + to_string(num_threads) + ">>>(" +
                        "(float (*)" + dims[0] + ") " + d_input_tensor->name() +
                        ", (float (*)" + dims[1] + ") " + d_output_tensor->name() + ");\n";
                break;
            }
            default:
                Assert(false);
        }
        run_iter_gen += "\n";

        const int num_prefilling_zeros = (int) ceil(log10(kernel_list.size())) - 
                to_string(kernel.kernel_id).size();
        // individual profiling, write everything to file separately for each of the
        // kernels
        if (is_individual) {
            cu_includes_gen.clear();
            tensor_declaration_gen.clear();
            main_addr_assign_gen.clear();
            main_data_gen_gen.clear();
            main_malloc_gen.clear();

            // on individual generation, kernel 0 is omitted
            if (kernel.kernel_id == 0) {
                run_iter_gen.clear();
                continue;
            }

            // if kernel need specific declaration, include declaration file that is
            // recorded in the kernel to definition filename reverse mapping, include
            // "cudannUtil.cuh" otherwise
            // for correct running of auto script, do not change this
            if (need_specific_declaration(kernel.type)) {
                cu_includes_gen += kernel_definition_filename[kernel.kernel_id] + "\n";
            } else {
                cu_includes_gen += "#include \"../include/cudadnnUtil.cuh\"\n";
                Assert(kernel_definition_filename.find(kernel.kernel_id) ==
                       kernel_definition_filename.end());
            }

            // declare all tensors for profiling
            // gather all tensors
            vector<Tensor *> all_tensors = {
                kernel.parent_layer->input_activation, kernel.parent_layer->output_activation,
                kernel.parent_layer->weight, kernel.parent_layer->d_input, kernel.parent_layer->d_output,
                kernel.parent_layer->d_weight, kernel.parent_layer->bias, kernel.parent_layer->d_bias,
                kernel.parent_layer->alpha_and_beta, kernel.parent_layer->d_alpha_and_beta,
                kernel.parent_layer->running_m, kernel.parent_layer->running_v,
                kernel.parent_layer->mu, kernel.parent_layer->var, kernel.parent_layer->v1,
                kernel.parent_layer->v2, kernel.parent_layer->d_mu, kernel.parent_layer->d_var,
                kernel.parent_layer->d_v1, kernel.parent_layer->d_v2, kernel.parent_layer->musk_array
            };
            all_tensors.insert(all_tensors.end(),
                  kernel.parent_layer->other_inputs.begin(),
                  kernel.parent_layer->other_inputs.end());
            all_tensors.insert(all_tensors.end(),
                  kernel.parent_layer->other_d_inputs.begin(),
                  kernel.parent_layer->other_d_inputs.end());
            all_tensors.insert(all_tensors.end(),
                  kernel.parent_layer->other_d_outputs.begin(),
                  kernel.parent_layer->other_d_outputs.end());
            // filter out required tensor
            vector<Tensor *> existing_tensors;
            for (Tensor *tensor : all_tensors)
                if (tensor) existing_tensors.push_back(tensor);
            tensor_declaration_gen = generate_tensor_declarations(existing_tensors,
                                                                  num_tensor_per_line);
            // generate helper function
            helper_func_gen = generate_helper_functions();

            // assign weight addresses according to the size of the tensor, generate
            // the data and mask accordingly
            main_addr_assign_gen += "  // Weight addr assignment\n";
            main_data_gen_gen += "  // Data & mask generation\n";
            long long total_tensor_size_byte = 0;
            for (Tensor *tensor : existing_tensors) {
                if (tensor == kernel.parent_layer->musk_array) {
                    Dropout *layer = dynamic_cast<Dropout *>(kernel.parent_layer->operatorr);
                    Assert(layer != nullptr);
                    main_data_gen_gen += "  generate_mask(" + kernel.parent_layer->musk_array->name() + ", " + 
                            to_string(tensor->size_in_byte / sizeof(float)) + ", " + 
                            to_string( layer->p) + ");\n";
                } else {
                    main_data_gen_gen += "  generate_data(" + tensor->name() + ", " + 
                            to_string(tensor->size_in_byte / sizeof(float)) + ");\n";
                }
                main_addr_assign_gen += "  " + tensor->name() + " = workspace_base + " + 
                        to_string(total_tensor_size_byte / sizeof(float)) + ";\n";
                total_tensor_size_byte += tensor->size_in_byte;
            }
            main_addr_assign_gen += "\n";
            main_data_gen_gen += "\n";

            main_malloc_gen += "  // Workspace malloc\n";
            if (is_UVM) {
                main_malloc_gen += "  cudaMallocManaged(&workspace_base, " + 
                        to_string(total_tensor_size_byte) + ");\n";
            } else {
                main_malloc_gen += "  cudaMalloc(&workspace_base, " + 
                        to_string(total_tensor_size_byte) + ");\n";
            }
            main_malloc_gen += "  \n";

            run_iter_gen = run_iter_prefix + run_iter_gen + run_iter_postfix1;
            run_iter_gen += string(num_prefilling_zeros, '0');
            run_iter_gen += to_string(kernel.kernel_id) + run_iter_postfix2;

            ofstream fout;
            string file = "./" + output_folder_name + "/profiling_src/profiling";
            run_iter_gen += string(num_prefilling_zeros, '0');
            file += to_string(kernel.kernel_id) + ".cu";

            fout.open(file);
            fout << cu_includes_gen;
            fout << file_begin_gen;
            fout << helper_func_gen;
            fout << tensor_declaration_gen;
            fout << run_iter_gen;
            fout << main_begin_gen;
            fout << main_malloc_gen;
            fout << main_addr_assign_gen;
            fout << main_data_gen_gen;
            fout << main_end_gen;
            fout.close();
            run_iter_gen.clear();
        } else {
            run_iter_gen.pop_back();
            run_iter_gen += "  cudaEventRecord(stop, 0);\n";
            run_iter_gen += "  CUDA_CHECK_RET( cudaEventSynchronize(stop) );\n";
            run_iter_gen += "  cudaEventElapsedTime(&elapsedTime, start, stop);\n";
            run_iter_gen += "  printf(\"";
            run_iter_gen += string(num_prefilling_zeros, '0') + to_string(kernel.kernel_id);
            run_iter_gen += " %20.10f ms\\n\", elapsedTime);\n\n";
        }
    }

    if (!is_individual) {
        // declare all tensors for non-profiling
        tensor_declaration_gen = generate_tensor_declarations(tensor_list, num_tensor_per_line);
        helper_func_gen = generate_helper_functions();

        run_iter_gen = run_iter_prefix + run_iter_gen + run_iter_postfix1;

        ofstream fout;
        string file = "./" + output_folder_name + "/main.cu";
        fout.open(file);
        fout << cu_includes_gen;
        fout << file_begin_gen;
        fout << helper_func_gen;
        fout << tensor_declaration_gen;
        fout << run_iter_gen;
        fout << main_begin_gen;
        fout << main_malloc_gen;
        fout << main_addr_assign_gen;
        fout << main_data_gen_gen;
        fout << main_end_gen;
        fout << "\n// TOTAL_KERNEL_NUM: " << kernel_list.size() << "\n";
        fout.close();
    }
    iprintf("Codegen done\n", "");
}


void cudnn_profiling(bool individual_run, bool workspace_only) {
    ofstream fout, fconfigout, fworkspaceout;
    string filename = "./" + output_folder_name + "/cudnn_kernel_times.txt";
    string config_filename = "./" + output_folder_name + "/cudnn_input_data.txt";
    string workspace_filename = "./" + output_folder_name + "/cudnn_workspace_sizes.txt";
    // write to profiling result file
    if (!workspace_only) {
      fout.open(filename, ofstream::out | ofstream::trunc);
      Assert(fout.good());
      fout.close();
      fout.open(filename, ofstream::app);
    }
    // write to workspace size file anyway
    fworkspaceout.open(workspace_filename, ofstream::out | ofstream::trunc);
    Assert(fworkspaceout.good());
    fworkspaceout.close();
    fworkspaceout.open(workspace_filename, ofstream::app);
    // write to config file anyway
    fconfigout.open(config_filename, ofstream::out | ofstream::trunc);
    Assert(fconfigout.good());
    fconfigout.close();
    fconfigout.open(config_filename, ofstream::app);

    vector<long> args;
    for (CUDAKernel kernel : kernel_list) {
        if (individual_run)
            iprintf("%d: ", kernel.kernel_id);
        args.clear();
        if (kernel.parent_layer)
        {
          switch (kernel.type) {
              case Conv2d_Forward:
              case Conv2d_Backward_Input:
              case Conv2d_Backward_Weight: {
                  Conv2d *layer = dynamic_cast<Conv2d *>(kernel.parent_layer->operatorr);
                  Assert(layer != nullptr);
                  args.push_back(kernel.parent_layer->N);
                  args.push_back(kernel.parent_layer->C);
                  args.push_back(kernel.parent_layer->H);
                  args.push_back(kernel.parent_layer->W);
                  args.push_back(layer->out_channels);
                  args.push_back(layer->kernel_size_r);
                  args.push_back(layer->kernel_size_s);
                  args.push_back(layer->padding_0);
                  args.push_back(layer->padding_1);
                  args.push_back(layer->stride_0);
                  args.push_back(layer->stride_1);
                  break;
              }
              case ReLU_Forward:
              case ReLU_Backward: {
                  args.push_back(kernel.parent_layer->N);
                  args.push_back(kernel.parent_layer->C);
                  args.push_back(kernel.parent_layer->H);
                  args.push_back(kernel.parent_layer->W);
                  break;
              }
              case MaxPool2d_Forward:
              case MaxPool2d_Backward: {
                  MaxPool2d *layer = dynamic_cast<MaxPool2d *>(kernel.parent_layer->operatorr);
                  Assert(layer != nullptr);
                  args.push_back(kernel.parent_layer->N);
                  args.push_back(kernel.parent_layer->C);
                  args.push_back(kernel.parent_layer->H);
                  args.push_back(kernel.parent_layer->W);
                  args.push_back(layer->kernel_size);
                  args.push_back(layer->kernel_size);
                  args.push_back(layer->padding);
                  args.push_back(layer->padding);
                  args.push_back(layer->stride);
                  args.push_back(layer->stride);
                  break;
              }
              case AdaptiveAvgPool2d_Forward:
              case AdaptiveAvgPool2d_Backward: {
                  AdaptiveAvgPool2d *layer = dynamic_cast<AdaptiveAvgPool2d *>(kernel.parent_layer->operatorr);
                  Assert(layer != nullptr);
                  args.push_back(kernel.parent_layer->N);
                  args.push_back(kernel.parent_layer->C);
                  args.push_back(kernel.parent_layer->H);
                  args.push_back(kernel.parent_layer->W);
                  args.push_back(kernel.parent_layer->H / layer->outputsize_0);
                  args.push_back(kernel.parent_layer->W / layer->outputsize_1);
                  args.push_back(0);
                  args.push_back(0);
                  args.push_back(layer->outputsize_0);
                  args.push_back(layer->outputsize_1);
                  break;
              }
              case Linear_Forward:
              case Linear_Backward_Input:
              case Linear_Backward_Weight:
              case Linear_Backward_Bias: {
                  Linear *layer = dynamic_cast<Linear *>(kernel.parent_layer->operatorr);
                  Assert(layer != nullptr);
                  args.push_back(kernel.parent_layer->N);
                  args.push_back(kernel.parent_layer->C);
                  args.push_back(kernel.parent_layer->H);
                  args.push_back(kernel.parent_layer->W);
                  args.push_back(layer->in_features);
                  args.push_back(layer->out_features);
                  break;
              }
              case BatchNorm2d_Forward:
              case BatchNorm2d_Backward: {
                  args.push_back(kernel.parent_layer->N);
                  args.push_back(kernel.parent_layer->C);
                  args.push_back(kernel.parent_layer->H);
                  args.push_back(kernel.parent_layer->W);
                  break;
              }
              case Add_Forward:
              case Add_MultiGredient: {
                  args.push_back(kernel.parent_layer->other_inputs.size() + 1);
                  args.push_back(kernel.parent_layer->output_activation->size_in_byte / sizeof(float));
                  args.push_back(kernel.parent_layer->N);
                  args.push_back(kernel.parent_layer->C);
                  break;
              }
              case Concat_Forward:
              case Concat_Backward: {
                  args.push_back(kernel.parent_layer->N);
                  args.push_back(kernel.parent_layer->H);
                  args.push_back(kernel.parent_layer->W);
                  args.push_back(kernel.parent_layer->input_Cs.size());
                  for (int input_c : kernel.parent_layer->input_Cs)
                    args.push_back(input_c);
                  break;
              }
              case Scale_Forward:
              case Scale_Backward: {
                  args.push_back(kernel.parent_layer->N);
                  args.push_back(kernel.parent_layer->C);
                  args.push_back(kernel.parent_layer->scale_H);
                  args.push_back(kernel.parent_layer->scale_W);
                  break;
              }
              case Dropout_Forward:
              case Dropout_Backward: {
                  args.push_back(kernel.parent_layer->N);
                  args.push_back(kernel.parent_layer->C);
                  args.push_back(kernel.parent_layer->H);
                  args.push_back(kernel.parent_layer->W);
                  break;
              }
              case Conv2d_Apply_Grad: {
                  args.push_back(kernel.parent_layer->weight->size_in_byte / sizeof(float));
                  args.push_back(kernel.parent_layer->N);
                  args.push_back(kernel.parent_layer->C);
                  break;
              }
              case Linear_Apply_Grad_Bias: {
                  args.push_back(kernel.parent_layer->bias->size_in_byte / sizeof(float));
                  args.push_back(kernel.parent_layer->N);
                  args.push_back(kernel.parent_layer->C);
                  break;
              }
              case Linear_Apply_Grad_Weight: {
                  args.push_back(kernel.parent_layer->weight->size_in_byte / sizeof(float));
                  args.push_back(kernel.parent_layer->N);
                  args.push_back(kernel.parent_layer->C);
                  break;
              }
              case BatchNorm2d_Apply_Grad: {
                  args.push_back(kernel.parent_layer->alpha_and_beta->size_in_byte / sizeof(float));
                  args.push_back(kernel.parent_layer->N);
                  args.push_back(kernel.parent_layer->C);
                  break;
              }
              case LoadData_A0: {
                  args.push_back((long) kernel.parent_layer->N * kernel.parent_layer->C * kernel.parent_layer->H * kernel.parent_layer->W);
                  args.push_back(kernel.parent_layer->N);
                  args.push_back(kernel.parent_layer->C);
                  break;
              }
              case makeLoss: {
                  args.push_back(kernel.parent_layer->output_activation->size_in_byte / sizeof(float));
                  args.push_back(kernel.parent_layer->N);
                  args.push_back(kernel.parent_layer->C);
                  break;
              }
              default:
                  Assert(false);
          }
        }
        else
        {
          switch (kernel.type) {
              case GatherV2_Forward:
              case GatherV2_Backward: {
                  Model_OP *op = kernel.parent_op;
                  Assert(op != nullptr);
                  args.push_back(op->output_tensor->size_in_byte / 4);
                  args.push_back(512);
                  args.push_back(1024);
                  break;
              }
              case ReLU_Forward:
              case ReLU_Backward:
              case SoftmaxBasic_Forward:
              case SoftmaxBasic_Backward: {
                  Model_OP *op = kernel.parent_op;
                  Assert(op != nullptr);
                  long values[4];
                  for (int i = 3; i >= 0; i--)
                  {
                    if (i<op->output_dims.size() && i>=0)
                    {
                      values[i] = op->output_dims[i];
                    }
                    else
                    {
                      values[i] = 1;
                    }
                    
                  }
                  args.push_back(values[0]);
                  args.push_back(values[1]);
                  args.push_back(values[2]);
                  args.push_back(values[3]);
                  break;
              }
              case Conv2d_Forward:
              case Conv2d_Backward_Input:
              case Conv2d_Backward_Weight: {
                  Model_OP *op = kernel.parent_op;
                  Assert(op != nullptr);
                  long values[4];
                  args.push_back(op->input_tensors[0].dims[0]);
                  args.push_back(op->input_tensors[0].dims[1]);
                  args.push_back(op->input_tensors[0].dims[2]);
                  args.push_back(op->input_tensors[0].dims[3]);
                  args.push_back(op->input_tensors[1].dims[0]);
                  args.push_back(op->input_tensors[1].dims[2]);
                  args.push_back(op->input_tensors[1].dims[3]);
                  args.push_back(0);
                  args.push_back(0);
                  args.push_back(32);
                  args.push_back(32);
                  break;
              }
              case Linear_Forward:
              case Linear_Backward_Input:
              case Linear_Backward_Weight:
              case Linear_Backward_Bias: {
                  Model_OP *op = kernel.parent_op;
                  Assert(op != nullptr);
                  long values[4];
                  for (int i = 3; i >= 0; i--)
                  {
                    if (i<op->input_tensors[0].dims.size() && i>=0)
                    {
                      values[i] = op->input_tensors[0].dims[i];
                    }
                    else
                    {
                      values[i] = 1;
                    }
                    
                  }
                  args.push_back(values[0]);
                  args.push_back(values[1]);
                  args.push_back(values[2]);
                  args.push_back(values[3]);
                  args.push_back(op->input_tensors[1].dims[0]);
                  args.push_back(op->input_tensors[1].dims[1]);
                  break;
              }
              case Add_Forward:
              case Add_Backward: 
              case Subtract_Forward:
              case Subtract_Backward:{
                  Model_OP *op = kernel.parent_op;
                  Assert(op != nullptr);
                  args.push_back(2);
                  args.push_back(op->output_tensor->size_in_byte / 4);
                  args.push_back(512);
                  args.push_back(1024);
                  break;
              }
              case Add_MultiGredient: {
                  Model_OP *op = kernel.parent_op;
                  Assert(op != nullptr);
                  args.push_back(op->d_output_tensors.size());
                  args.push_back(op->d_output_tensors[0]->size_in_byte / 4);
                  args.push_back(512);
                  args.push_back(1024);
                  break;
              }
              case BatchMatMul_Forward:
              case BatchMatMul_Backward: {
                  Model_OP *op = kernel.parent_op;
                  Assert(op != nullptr);
                  long values[4];
                  for (int i = 3; i >= 0; i--)
                  {
                    if (i<op->input_tensors[0].dims.size() && i>=0)
                    {
                      values[i] = op->input_tensors[0].dims[i];
                    }
                    else
                    {
                      values[i] = 1;
                    }
                    
                  }
                  args.push_back(values[0]);
                  args.push_back(values[1]);
                  args.push_back(values[2]);
                  args.push_back(values[3]);
                  args.push_back(op->input_tensors[1].dims[2]);
                  args.push_back(op->input_tensors[1].dims[3]);
                  break;
              }
              case Divide_Forward:
              case Divide_Backward_A:
              case Divide_Backward_B:
              case Multiply_Forward:
              case Multiply_Backward:
              case Tanh_Forward:
              case Tanh_Backward:
              case Erf_Forward:
              case Erf_Backward:
              case Power_Forward:
              case Power_Backward: {
                  Model_OP *op = kernel.parent_op;
                  Assert(op != nullptr);
                  args.push_back(op->output_tensor->size_in_byte / 4);
                  args.push_back(512);
                  args.push_back(1024);
                  break;
              }
              case Sqrt_Forward:
              case Sqrt_Backward: {
                  Model_OP *op = kernel.parent_op;
                  Assert(op != nullptr);
                  args.push_back(op->input_tensors[0].tensor->size_in_byte / 4);
                  args.push_back(512);
                  args.push_back(1024);
                  break;
              }
              case Sum_Forward:
              case Sum_Backward: {
                  Model_OP *op = kernel.parent_op;
                  Assert(op != nullptr);
                  args.push_back(op->output_tensor->size_in_byte / 4);
                  args.push_back(512);
                  args.push_back(1024);
                  args.push_back(op->input_tensors[0].tensor->size_in_byte / op->output_tensor->size_in_byte);
                  break;
              }
              case Apply_Grad:
              case Linear_Apply_Grad_Weight: {
                  auto it = kernel.outputs.begin();
                  long val = (*it)->size_in_byte / 4;
                  args.push_back(val);
                  args.push_back(512);
                  args.push_back(1024);
                  break;
              }
              case makeLoss: {
                  auto it = kernel.outputs.begin();
                  long val = (*it)->size_in_byte / 4;
                  args.push_back(val);
                  args.push_back(512);
                  args.push_back(1024);
                  break;
              }
              default:
                  Assert(false);
          }
        }
        
        
        
        if (individual_run) {
            if (kernel.type != LoadData_A0) {
                if (is_input_pf_only) {
                    args.push_back(1);
                    args.push_back(0);
                } else if (is_UVM) {
                    args.push_back(0);
                    args.push_back(0);
                } else {
                    args.push_back(1);
                    args.push_back(1);
                }
            }
            string result;
            const int num_prefilling_zeros = (int) ceil(log10(kernel_list.size())) - to_string(kernel.kernel_id).size();
            // kernel times
            if (!workspace_only) {
                result = exec(kernel.type, is_UVM, args);
                fout << string(num_prefilling_zeros, '0') << to_string(kernel.kernel_id) << " " << result << "\n";
                fout.flush();
            }
            // kernel workspace
            result = exec(kernel.type, 2, args);
            fworkspaceout << string(num_prefilling_zeros, '0') << to_string(kernel.kernel_id) << " " << result << "\n";
            fworkspaceout.flush();
        }
        fconfigout << print_kerneltype_array[kernel.type] << " ";
        for (long arg : args)
            fconfigout << arg << " ";
            
        if (kernel.type != LoadData_A0) {
            if (is_input_pf_only) {
                fconfigout << "1 0 ";
            } else if (is_UVM) {
                fconfigout << "0 0 ";
            } else {
                fconfigout << "1 1 ";
            }
        }
        fconfigout << "\n";
        fconfigout.flush();
    }
    fconfigout.close();
    iprintf("CUDNN profile input file have been saved to <%s>\n", config_filename.c_str());
    if (!individual_run) {
        string result;
        iprintf("CUDNN workspace start grouped run\n", "");
        result = exec(config_filename, 2);
        fworkspaceout << result;
        if (!workspace_only) {
            iprintf("CUDNN profile start grouped run\n", "");
            result = exec(config_filename, is_UVM);
            fout << result;
        }
        fworkspaceout.close();
        iprintf("CUDNN workspace size file have been saved to <%s>\n", workspace_filename.c_str());
    } else {
        fworkspaceout.close();
        iprintf("CUDNN workspace size file have been saved to <%s>\n", workspace_filename.c_str());
    }
    if (!workspace_only) {
        fout.close();
    }
}

















// long main () {
//   cout << conv2d_forward_CodeGen(1, 28, 28, 64, 1, 6, 7, 7, 1, 1, 2, 2, true);
//   cout << conv2d_backward_weight_CodeGen(1, 28, 28, 64, 1, 6, 7, 7, 1, 1, 2, 2, true);
//   cout << conv2d_backward_input_CodeGen(1, 28, 28, 64, 1, 6, 7, 7, 1, 1, 2, 2, true);
//   cout << maxPool2d_forward_CodeGen(2, 64, 6, 24, 24, 3, 3, 3, 0, 1, false);
//   cout << maxPool2d_backward_CodeGen(2, 64, 6, 24, 24, 3, 3, 3, 0, 1, false);
//   cout << reLU_forward_CodeGen(3, 64, 6, 24, 24);
//   cout << reLU_backward_CodeGen(3, 64, 6, 24, 24);
//   cout << avgAdaptivePool2d_forward_CodeGen(4, 64, 6, 24, 24, 8, 8);
//   cout << avgAdaptivePool2d_backward_CodeGen(4, 64, 6, 24, 24, 8, 8);

//   cout << linear_forward_CodeGen(5, 64, 128, 1, 1, 128, 200);
//   cout << linear_backward_bias_CodeGen(5, 64, 128, 1, 1, 128, 200);
//   cout << linear_backward_weight_CodeGen(5, 64, 128, 1, 1, 128, 200);
//   cout << linear_backward_input_CodeGen(5, 64, 128, 1, 1, 128, 200);

//   cout << batchnorm2d_forward_CodeGen(5, 64, 6, 24, 24, 0.00001, 0.1, true);
//   cout << batchnorm2d_backward_CodeGen(5, 64, 6, 24, 24);
//   cout << add_forward_CodeGen(10, 30000);
//   cout << multigradient_add_CodeGen(10, 30000);
// }
