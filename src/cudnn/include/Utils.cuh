#pragma once
#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <vector>

#define CUDNN_CALL(func) {                                                                         \
  cudnnStatus_t status = (func);                                                                   \
  if (status != CUDNN_STATUS_SUCCESS) {                                                            \
    std::cerr << "Error in file " << __FILE__ << " on line " << __LINE__ << ": " << cudnnGetErrorString(status) << std::endl; \
    std::exit(EXIT_FAILURE);                                                                       \
  }                                                                                                \
}

#define CUDA_CALL(func) {                                                                           \
  cudaError_t status = (func);                                                                      \
  if (status != cudaSuccess) {                                                                         \
    std::cerr << "Error in file " << __FILE__ << " on line " << __LINE__ << ": " << cudaGetErrorString(status) << std::endl;  \
    std::exit(1);                                                                                   \
  }                                                                                                 \
}

class Profiler {
  public:
    virtual ~Profiler() {};
    virtual float Run() = 0;
    virtual size_t getWorkspaceSize() { return 0; };
};

void GPUInitRand();
void GPUFillRand(float *target, long size);
void CPUFillRand(float *target, long size);

__global__ void gpu_access(float *target, long size);