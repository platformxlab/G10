#include <curand.h>
#include "Utils.cuh"

#define RANDOM_INIT

curandGenerator_t prng;

void GPUInitRand() {
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
}

void GPUFillRand(float *target, long size) {
#ifdef RANDOM_INIT
    curandGenerateUniform(prng, target, size / sizeof(float));
#endif
}

void CPUFillRand(float *target, long size) {
#ifdef RANDOM_INIT
  for(long i = 0; i < size / sizeof(float); i++) {
    target[i] = (float) (rand() % 256); 
  }
#endif
}


__global__ void gpu_access(float *target, long size){
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	if(thread_pos > 0 && thread_pos < size){
    target[thread_pos]++;
  }
}
