#include <cstdlib>
#include <vector>
#include <memory>


#ifndef CUDADNNUTIL_H
#define CUDADNNUTIL_H



const static float dt = 1.0E-01f;
const static float threshold = 1.0E-02f;//TDOD:?


// Utility CUDA kernel functions
__global__ void forwardPass_Dropout(float* input, float* output, float* musk, int N);
__global__ void backwardPass_Dropout(float* d_output, float* d_input, float* musk, int N);
__global__ void makeError(float *err, float *output, unsigned int Y, const int N);
__global__ void apply_grad(float *output, float *grad, const int N);
__global__ void transfer_A0(float* data, float* activation_tensor, const int N);







#endif
