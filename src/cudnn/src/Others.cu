#include <assert.h>
#include <vector>
#include "Others.h"

__global__ void add(float **input_data, float *output_data, int num_inputs, long N) {
    const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const long parallel_size = blockDim.x * gridDim.x;
    
    for (long n = 0; n < N; n += parallel_size) {
        long idx = n + thread_pos;
        if (idx < N) {
            for(long i = 0; i < num_inputs; i++) {
                output_data[idx] += input_data[i][idx];
            }
        }
    }
}

__global__ void add_backward(float **output_data, float *input_data, int num_inputs, long N) {
    const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const long parallel_size = blockDim.x * gridDim.x;
    
    for (long n = 0; n < N; n += parallel_size) {
        long idx = n + thread_pos;
        if (idx < N) {
            for(long i = 0; i < num_inputs; i++) {
                output_data[i][idx] = input_data[idx];
            }
        }
    }
}


__global__ void gather(float *output_data, float *input_data, long N) {
    const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const long parallel_size = blockDim.x * gridDim.x;
    
    for (long n = 0; n < N; n += parallel_size) {
        long idx = n + thread_pos;
        if (idx < N) {
            output_data[idx] = input_data[idx];
        }
    }
}



__global__ void divide(float *output_data, float *input_dataA, float *input_dataB, long N) {
    const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const long parallel_size = blockDim.x * gridDim.x;
    
    for (long n = 0; n < N; n += parallel_size) {
        long idx = n + thread_pos;
        if (idx < N) {
            output_data[idx] = input_dataA[idx] / input_dataB[idx];
        }
    }
}


__global__ void divide_backwardA(float *da, float *b, float *dc, long N) {
    const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const long parallel_size = blockDim.x * gridDim.x;
    
    for (long n = 0; n < N; n += parallel_size) {
        long idx = n + thread_pos;
        if (idx < N) {
            da[idx] = b[idx] * dc[idx];

        }
    }
}


__global__ void divide_backwardB(float *db, float *b, float *c, float *dc, long N) {
    const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const long parallel_size = blockDim.x * gridDim.x;
    
    for (long n = 0; n < N; n += parallel_size) {
        long idx = n + thread_pos;
        if (idx < N) {
            db[idx] = - b[idx] / c[idx] *dc[idx];
        }
    }
}


__global__ void multiply(float *output_data, float *input_dataA, float *input_dataB, long N) {
    const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const long parallel_size = blockDim.x * gridDim.x;
    
    for (long n = 0; n < N; n += parallel_size) {
        long idx = n + thread_pos;
        if (idx < N) {
            output_data[idx] = input_dataA[idx] * input_dataB[idx];
        }
    }
}

//Mult_backward: divide


__global__ void power_forward(float *output_data, float *input_dataA, float *input_dataB, long N) {
    const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const long parallel_size = blockDim.x * gridDim.x;
    
    for (long n = 0; n < N; n += parallel_size) {
        long idx = n + thread_pos;
        if (idx < N) {
            output_data[idx] = pow(input_dataA[idx], input_dataB[idx]);
        }
    }
}


__global__ void power_backward(float *da, float *a, float *b, float *dc, long N) {
    const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const long parallel_size = blockDim.x * gridDim.x;
    
    for (long n = 0; n < N; n += parallel_size) {
        long idx = n + thread_pos;
        if (idx < N) {
            da[idx] = dc[idx] / (b[idx] * a[idx]);
        }
    }
}

__global__ void sqrt_forward(float *output, float *input, long N) {
    const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const long parallel_size = blockDim.x * gridDim.x;
    
    for (long n = 0; n < N; n += parallel_size) {
        long idx = n + thread_pos;
        if (idx < N) {
            output[idx] = sqrt(input[idx]);
        }
    }
}


__global__ void sqrt_backward(float *da, float *a, float *db, long N) {
    const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const long parallel_size = blockDim.x * gridDim.x;
    
    for (long n = 0; n < N; n += parallel_size) {
        long idx = n + thread_pos;
        if (idx < N) {
            da[idx] = 2 *db[idx] / (1 / a[idx] * sqrt(a[idx]));
        }
    }
}

__global__ void sum_forward(float *a, float *b, long dim_3, long Nb) {
    const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const long parallel_size = blockDim.x * gridDim.x;
    
    for (long n = 0; n < Nb; n += parallel_size) {
        long idx = n + thread_pos;
        if (idx < Nb) {
            b[idx] = 0;
            for(long i = 0; i < dim_3; i++){
                b[idx] += a[idx*dim_3 + i];
            }
        }
    }
}

__global__ void sum_backward(float *da, float *db, long dim_3, long Nb) {
    const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const long parallel_size = blockDim.x * gridDim.x;
    
    for (long n = 0; n < Nb; n += parallel_size) {
        long idx = n + thread_pos;
        if (idx < Nb) {
            da[idx] = 0;
            for(long i = 0; i < dim_3; i++){
                da[idx*dim_3 + i] = db[idx];
            }
        }
    }
}

__global__ void tanh_forward(float *output, float *input, long N){

    const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const long parallel_size = blockDim.x * gridDim.x;
    
    for (long n = 0; n < N; n += parallel_size) {
        long idx = n + thread_pos;
        if (idx < N) {
            output[idx] = tanh(input[idx]);
        }
    }
}


__global__ void tanh_backward(float *dx, float *x, float *dy, long N){

    const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const long parallel_size = blockDim.x * gridDim.x;
    
    for (long n = 0; n < N; n += parallel_size) {
        long idx = n + thread_pos;
        if (idx < N) {
            dx[idx] = dy[idx] / (1 - pow((tanh(x[idx])), 2));
        }
    }
}


__global__ void erf_forward(float *output, float *input, long N){

    const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const long parallel_size = blockDim.x * gridDim.x;
    
    for (long n = 0; n < N; n += parallel_size) {
        long idx = n + thread_pos;
        if (idx < N) {
            output[idx] = erf(input[idx]);
        }
    }
}


__global__ void erf_backward(float *dx, float *x, float *dy, long N){

    const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const long parallel_size = blockDim.x * gridDim.x;
    
    for (long n = 0; n < N; n += parallel_size) {
        long idx = n + thread_pos;
        if (idx < N) {
            dx[idx] = dy[idx] / (exp(-0.5*x[idx]));
        }
    }
}


__global__ void apply_grad(float *output, float *grad, const long N) {
	const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int parallel_size = blockDim.x * gridDim.x;

    for(int n = 0; n < N; n += parallel_size){
        int idx = n + thread_pos;
        if(idx < N) {
            output[idx] += dt * grad[idx];
        }
    }
}

__global__ void makeError(float *err, float *output, unsigned int Y, const long N) {
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = N * pos / size; idx < N * (pos + 1) / size; ++idx) {
        if(idx < N) {
		    err[idx] = ((Y == idx ? 1.0f : 0.0f) - output[idx]);
        }
	}
}




Add_Backward::Add_Backward(cudnnHandle_t handle, vector<double> &args, bool is_UVM) : 
        handle(handle), is_UVM(is_UVM) {
    // 0. #inputs     1. length     2. batch_size   3. num_threads
    num_input  = args[0];   N          = args[1];
    batch_size = args[2];   num_threads= args[3];
    input_ratio = args[4]; output_ratio = args[5];

    cudaMalloc(&device_d_input_data, (long) num_input * sizeof(float *));
    d_input_data = (float **) malloc(num_input * sizeof(float *));
    // Alloc
    if (!is_UVM) {
        for (int i = 0; i < num_input; i++) {
            CUDA_CALL(cudaMalloc(&d_input_data[i], (long) N * sizeof(float)));
            //GPUFillRand(input_data[i], (long) N * sizeof(float));
        }
        CUDA_CALL(cudaMalloc(&d_output_data, (long) N * sizeof(float)));
        CUDA_CALL(cudaMemcpy(device_d_input_data, d_input_data, (long) num_input * sizeof(float *), cudaMemcpyHostToDevice));
    }
    cudaDeviceSynchronize();
}

Add_Backward::~Add_Backward() {
    if (!is_UVM) {
        for (int i = 0; i < num_input; i++)
            CUDA_CALL(cudaFree(d_input_data[i]));
        CUDA_CALL(cudaFree(d_output_data));
    }
    CUDA_CALL(cudaFree(device_d_input_data));
    free(d_input_data);
}

float Add_Backward::Run() {
    if (is_UVM) {
        for (int i = 0; i < num_input; i++) {
            CUDA_CALL(cudaMallocManaged(&d_input_data[i], (long) N * sizeof(float)));
            CPUFillRand(d_input_data[i], (long) N * sizeof(float));
            GPUFillRand(d_input_data[i], (long) N * sizeof(float) * output_ratio);
        }
        CUDA_CALL(cudaMallocManaged(&d_output_data, (long) N * sizeof(float)));
        CPUFillRand(d_output_data, (long) N * sizeof(float));
        GPUFillRand(d_output_data, (long) N * sizeof(float) * input_ratio);
        CUDA_CALL(cudaMemcpy(device_d_input_data, d_input_data, (long) num_input * sizeof(float *), cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();
    }

    float milliseconds = 0;
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));

    CUDA_CALL(cudaEventRecord(start));
    add_backward<<<batch_size, num_threads>>>(device_d_input_data, d_output_data, num_input, N);
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    
    if (is_UVM) {
        for (int i = 0; i < num_input; i++)
            CUDA_CALL(cudaFree(d_input_data[i]));
        CUDA_CALL(cudaFree(d_output_data));
    }
    return milliseconds;
}




Add::Add(cudnnHandle_t handle, vector<double> &args, bool is_UVM) : 
        handle(handle), is_UVM(is_UVM) {
    // 0. #inputs     1. length     2. batch_size   3. num_threads
    num_input  = args[0];   N          = args[1];
    batch_size = args[2];   num_threads= args[3];
    input_ratio = args[4]; output_ratio = args[5];

    cudaMalloc(&device_input_data, (long) num_input * sizeof(float *));
    input_data = (float **) malloc(num_input * sizeof(float *));
    // Alloc
    if (!is_UVM) {
        for (int i = 0; i < num_input; i++) {
            CUDA_CALL(cudaMalloc(&input_data[i], (long) N * sizeof(float)));
            GPUFillRand(input_data[i], (long) N * sizeof(float));
        }
        CUDA_CALL(cudaMalloc(&output_data, (long) N * sizeof(float)));
        CUDA_CALL(cudaMemcpy(device_input_data, input_data, (long) num_input * sizeof(float *), cudaMemcpyHostToDevice));
    }
    cudaDeviceSynchronize();
}

Add::~Add() {
    if (!is_UVM) {
        for (int i = 0; i < num_input; i++)
            CUDA_CALL(cudaFree(input_data[i]));
        CUDA_CALL(cudaFree(output_data));
    }
    CUDA_CALL(cudaFree(device_input_data));
    free(input_data);
}

float Add::Run() {
    if (is_UVM) {
        for (int i = 0; i < num_input; i++) {
            CUDA_CALL(cudaMallocManaged(&input_data[i], (long) N * sizeof(float)));
            CPUFillRand(input_data[i], (long) N * sizeof(float));
            GPUFillRand(input_data[i], (long) N * sizeof(float) * input_ratio);
        }
        CUDA_CALL(cudaMallocManaged(&output_data, (long) N * sizeof(float)));
        CPUFillRand(output_data, (long) N * sizeof(float));
        GPUFillRand(output_data, (long) N * sizeof(float) * output_ratio);
        CUDA_CALL(cudaMemcpy(device_input_data, input_data, (long) num_input * sizeof(float *), cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();
    }

    float milliseconds = 0;
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));

    CUDA_CALL(cudaEventRecord(start));
    add<<<batch_size, num_threads>>>(device_input_data, output_data, num_input, N);
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    
    if (is_UVM) {
        for (int i = 0; i < num_input; i++)
            CUDA_CALL(cudaFree(input_data[i]));
        CUDA_CALL(cudaFree(output_data));
    }
    return milliseconds;
}





Divide_Forward::Divide_Forward(cudnnHandle_t handle, vector<double> &args, bool is_UVM) : 
        handle(handle), is_UVM(is_UVM) {
    // 0. length     1. batch_size   2. num_threads
    N          = args[0];   batch_size = args[1];   num_threads= args[2];
    input_ratio = args[3]; output_ratio = args[4];
    // Alloc
    if (!is_UVM) {
        CUDA_CALL(cudaMalloc(&input_a, (long) N * sizeof(float)));
        CUDA_CALL(cudaMalloc(&input_b, (long) N * sizeof(float)));
        CUDA_CALL(cudaMalloc(&output_c, (long) N * sizeof(float)));
        GPUFillRand(input_a, (long) N * sizeof(float));
        GPUFillRand(input_b, (long) N * sizeof(float));
        GPUFillRand(output_c, (long) N * sizeof(float));
    }
}

Divide_Forward::~Divide_Forward() {
    if (!is_UVM) {
        CUDA_CALL(cudaFree(input_a));
        CUDA_CALL(cudaFree(input_b));
        CUDA_CALL(cudaFree(output_c));
    }
}

float Divide_Forward::Run() {
    if (is_UVM) {
        CUDA_CALL(cudaMallocManaged(&input_a, (long) N * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&input_b, (long) N * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&output_c, (long) N * sizeof(float)));
        CPUFillRand(input_a, (long) N * sizeof(float));
        CPUFillRand(input_b, (long) N * sizeof(float));

        GPUFillRand(output_c, (long) N * sizeof(float) * output_ratio);
        GPUFillRand(input_a, (long) N * sizeof(float) * input_ratio);
        GPUFillRand(input_b, (long) N * sizeof(float) * input_ratio);
        cudaDeviceSynchronize();
    }

    float milliseconds = 0;
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    
    CUDA_CALL(cudaEventRecord(start));
    divide<<<batch_size, num_threads>>>(output_c, input_a, input_b, N);
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    

    return milliseconds;
}





Divide_Backward_A::Divide_Backward_A(cudnnHandle_t handle, vector<double> &args, bool is_UVM) : 
        handle(handle), is_UVM(is_UVM) {
    // 0. length     1. batch_size   2. num_threads
    N          = args[0];   batch_size = args[1];   num_threads= args[2];
    input_ratio = args[3]; output_ratio = args[4];
    // Alloc
    if (!is_UVM) {
        CUDA_CALL(cudaMalloc(&da, (long) N * sizeof(float)));
        CUDA_CALL(cudaMalloc(&b, (long) N * sizeof(float)));
        CUDA_CALL(cudaMalloc(&dc, (long) N * sizeof(float)));
        GPUFillRand(da, (long) N * sizeof(float));
        GPUFillRand(b, (long) N * sizeof(float));
        GPUFillRand(dc, (long) N * sizeof(float));
    }
}

Divide_Backward_A::~Divide_Backward_A() {
    if (!is_UVM) {
        CUDA_CALL(cudaFree(da));
        CUDA_CALL(cudaFree(b));
        CUDA_CALL(cudaFree(dc));
    }
}

float Divide_Backward_A::Run() {
    if (is_UVM) {
        CUDA_CALL(cudaMallocManaged(&da, (long) N * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&b, (long) N * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&dc, (long) N * sizeof(float)));
        CPUFillRand(dc, (long) N * sizeof(float));
        CPUFillRand(b, (long) N * sizeof(float));

        GPUFillRand(dc, (long) N * sizeof(float) * input_ratio);
        GPUFillRand(da, (long) N * sizeof(float) * output_ratio);
        GPUFillRand(b, (long) N * sizeof(float) * input_ratio);
        cudaDeviceSynchronize();
    }

    float milliseconds = 0;
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    
    CUDA_CALL(cudaEventRecord(start));
    divide_backwardA<<<batch_size, num_threads>>>(da, b, dc, N);
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    
    return milliseconds;
}




Divide_Backward_B::Divide_Backward_B(cudnnHandle_t handle, vector<double> &args, bool is_UVM) : 
        handle(handle), is_UVM(is_UVM) {
    // 0. length     1. batch_size   2. num_threads
    N          = args[0];   batch_size = args[1];   num_threads= args[2];
    input_ratio = args[3]; output_ratio = args[4];
    // Alloc
    if (!is_UVM) {
        CUDA_CALL(cudaMalloc(&db, (long) N * sizeof(float)));
        CUDA_CALL(cudaMalloc(&b, (long) N * sizeof(float)));
        CUDA_CALL(cudaMalloc(&c, (long) N * sizeof(float)));
        CUDA_CALL(cudaMalloc(&dc, (long) N * sizeof(float)));
        GPUFillRand(db, (long) N * sizeof(float));
        GPUFillRand(b, (long) N * sizeof(float));
        GPUFillRand(c, (long) N * sizeof(float));
        GPUFillRand(dc, (long) N * sizeof(float));
    }
}

Divide_Backward_B::~Divide_Backward_B() {
    if (!is_UVM) {
        CUDA_CALL(cudaFree(db));
        CUDA_CALL(cudaFree(b));
        CUDA_CALL(cudaFree(c));
        CUDA_CALL(cudaFree(dc));
    }
}

float Divide_Backward_B::Run() {
    if (is_UVM) {
        CUDA_CALL(cudaMallocManaged(&db, (long) N * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&b, (long) N * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&c, (long) N * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&dc, (long) N * sizeof(float)));
        CPUFillRand(dc, (long) N * sizeof(float));
        CPUFillRand(b, (long) N * sizeof(float));
        CPUFillRand(c, (long) N * sizeof(float));

        GPUFillRand(dc, (long) N * sizeof(float) * input_ratio);
        GPUFillRand(db, (long) N * sizeof(float) * output_ratio);
        GPUFillRand(b, (long) N * sizeof(float) * input_ratio);
        GPUFillRand(c, (long) N * sizeof(float) * input_ratio);
        cudaDeviceSynchronize();
    }

    float milliseconds = 0;
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    
    CUDA_CALL(cudaEventRecord(start));
    divide_backwardB<<<batch_size, num_threads>>>(db, b, c, dc, N);
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    
    return milliseconds;
}




Multiply::Multiply(cudnnHandle_t handle, vector<double> &args, bool is_UVM) : 
        handle(handle), is_UVM(is_UVM) {
    // 0. length     1. batch_size   2. num_threads
    N          = args[0];   batch_size = args[1];   num_threads= args[2];
    input_ratio = args[3]; output_ratio = args[4];
    // Alloc
    if (!is_UVM) {
        CUDA_CALL(cudaMalloc(&output, (long) N * sizeof(float)));
        CUDA_CALL(cudaMalloc(&inputA, (long) N * sizeof(float)));
        CUDA_CALL(cudaMalloc(&inputB, (long) N * sizeof(float)));
        GPUFillRand(output, (long) N * sizeof(float));
        GPUFillRand(inputA, (long) N * sizeof(float));
        GPUFillRand(inputB, (long) N * sizeof(float));
    }
}

Multiply::~Multiply() {
    if (!is_UVM) {
        CUDA_CALL(cudaFree(output));
        CUDA_CALL(cudaFree(inputA));
        CUDA_CALL(cudaFree(inputB));
    }
}

float Multiply::Run() {
    if (is_UVM) {
        CUDA_CALL(cudaMallocManaged(&output, (long) N * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&inputA, (long) N * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&inputB, (long) N * sizeof(float)));
        CPUFillRand(inputB, (long) N * sizeof(float));
        CPUFillRand(inputA, (long) N * sizeof(float));

        GPUFillRand(inputB, (long) N * sizeof(float) * input_ratio);
        GPUFillRand(output, (long) N * sizeof(float) * output_ratio);
        GPUFillRand(inputA, (long) N * sizeof(float) * input_ratio);
        cudaDeviceSynchronize();
    }

    float milliseconds = 0;
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    
    CUDA_CALL(cudaEventRecord(start));
    multiply<<<batch_size, num_threads>>>(output, inputA, inputB, N);
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    
    return milliseconds;
}



Power_Forward::Power_Forward(cudnnHandle_t handle, vector<double> &args, bool is_UVM) : 
        handle(handle), is_UVM(is_UVM) {
    // 0. length     1. batch_size   2. num_threads
    N          = args[0];   batch_size = args[1];   num_threads= args[2];
    input_ratio = args[3]; output_ratio = args[4];
    // Alloc
    if (!is_UVM) {
        CUDA_CALL(cudaMalloc(&output, (long) N * sizeof(float)));
        CUDA_CALL(cudaMalloc(&inputA, (long) N * sizeof(float)));
        CUDA_CALL(cudaMalloc(&inputB, (long) N * sizeof(float)));
        GPUFillRand(output, (long) N * sizeof(float));
        GPUFillRand(inputA, (long) N * sizeof(float));
        GPUFillRand(inputB, (long) N * sizeof(float));
    }
}

Power_Forward::~Power_Forward() {
    if (!is_UVM) {
        CUDA_CALL(cudaFree(output));
        CUDA_CALL(cudaFree(inputA));
        CUDA_CALL(cudaFree(inputB));
    }
}

float Power_Forward::Run() {
    if (is_UVM) {
        CUDA_CALL(cudaMallocManaged(&output, (long) N * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&inputA, (long) N * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&inputB, (long) N * sizeof(float)));
        CPUFillRand(inputB, (long) N * sizeof(float));
        CPUFillRand(inputA, (long) N * sizeof(float));

        GPUFillRand(inputB, (long) N * sizeof(float) * input_ratio);
        GPUFillRand(output, (long) N * sizeof(float) * output_ratio);
        GPUFillRand(inputA, (long) N * sizeof(float) * input_ratio);
        cudaDeviceSynchronize();
    }

    float milliseconds = 0;
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    
    CUDA_CALL(cudaEventRecord(start));
    power_forward<<<batch_size, num_threads>>>(output, inputA, inputB, N);
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    
    return milliseconds;
}









Power_Backward::Power_Backward(cudnnHandle_t handle, vector<double> &args, bool is_UVM) : 
        handle(handle), is_UVM(is_UVM) {
    // 0. length     1. batch_size   2. num_threads
    N          = args[0];   batch_size = args[1];   num_threads= args[2];
    input_ratio = args[3]; output_ratio = args[4];
    // Alloc
    if (!is_UVM) {
        CUDA_CALL(cudaMalloc(&da, (long) N * sizeof(float)));
        CUDA_CALL(cudaMalloc(&inputA, (long) N * sizeof(float)));
        CUDA_CALL(cudaMalloc(&inputB, (long) N * sizeof(float)));
        CUDA_CALL(cudaMalloc(&dc, (long) N * sizeof(float)));
        GPUFillRand(da, (long) N * sizeof(float));
        GPUFillRand(inputA, (long) N * sizeof(float));
        GPUFillRand(inputB, (long) N * sizeof(float));
        GPUFillRand(dc, (long) N * sizeof(float));
    }
}

Power_Backward::~Power_Backward() {
    if (!is_UVM) {
        CUDA_CALL(cudaFree(da));
        CUDA_CALL(cudaFree(inputA));
        CUDA_CALL(cudaFree(inputB));
        CUDA_CALL(cudaFree(dc));
    }
}

float Power_Backward::Run() {
    if (is_UVM) {
        CUDA_CALL(cudaMallocManaged(&da, (long) N * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&inputA, (long) N * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&inputB, (long) N * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&dc, (long) N * sizeof(float)));
        CPUFillRand(dc, (long) N * sizeof(float));
        CPUFillRand(inputA, (long) N * sizeof(float));
        CPUFillRand(inputB, (long) N * sizeof(float));

        GPUFillRand(dc, (long) N * sizeof(float) * input_ratio);
        GPUFillRand(da, (long) N * sizeof(float) * output_ratio);
        GPUFillRand(inputA, (long) N * sizeof(float) * input_ratio);
        GPUFillRand(inputB, (long) N * sizeof(float) * input_ratio);
        cudaDeviceSynchronize();
    }

    float milliseconds = 0;
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    
    CUDA_CALL(cudaEventRecord(start));
    power_backward<<<batch_size, num_threads>>>(da, inputA, inputB, dc, N);
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    
    return milliseconds;
}







Sqrt_Forward::Sqrt_Forward(cudnnHandle_t handle, vector<double> &args, bool is_UVM) : 
        handle(handle), is_UVM(is_UVM) {
    // 0. length     1. batch_size   2. num_threads
    N          = args[0];   batch_size = args[1];   num_threads= args[2];
    input_ratio = args[3]; output_ratio = args[4];
    // Alloc
    if (!is_UVM) {
        CUDA_CALL(cudaMalloc(&output, (long) N * sizeof(float)));
        CUDA_CALL(cudaMalloc(&input, (long) N * sizeof(float)));

        GPUFillRand(output, (long) N * sizeof(float));
        GPUFillRand(input, (long) N * sizeof(float));

    }
}

Sqrt_Forward::~Sqrt_Forward() {
    if (!is_UVM) {

        CUDA_CALL(cudaFree(output));
        CUDA_CALL(cudaFree(input));

    }
}

float Sqrt_Forward::Run() {
    if (is_UVM) {

        CUDA_CALL(cudaMallocManaged(&output, (long) N * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&input, (long) N * sizeof(float)));

        CPUFillRand(input, (long) N * sizeof(float));

        GPUFillRand(output, (long) N * sizeof(float) * output_ratio);
        GPUFillRand(input, (long) N * sizeof(float) * input_ratio);
        cudaDeviceSynchronize();
    }

    float milliseconds = 0;
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    
    CUDA_CALL(cudaEventRecord(start));
    sqrt_forward<<<batch_size, num_threads>>>(output, input, N);
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    
    return milliseconds;
}






Sqrt_Backward::Sqrt_Backward(cudnnHandle_t handle, vector<double> &args, bool is_UVM) : 
        handle(handle), is_UVM(is_UVM) {
    // 0. length     1. batch_size   2. num_threads
    N          = args[0];   batch_size = args[1];   num_threads= args[2];
    input_ratio = args[3]; output_ratio = args[4];
    // Alloc
    if (!is_UVM) {
        CUDA_CALL(cudaMalloc(&output_da, (long) N * sizeof(float)));
        CUDA_CALL(cudaMalloc(&input_a, (long) N * sizeof(float)));
        CUDA_CALL(cudaMalloc(&input_db, (long) N * sizeof(float)));

        GPUFillRand(output_da, (long) N * sizeof(float));
        GPUFillRand(input_a, (long) N * sizeof(float));
        GPUFillRand(input_db, (long) N * sizeof(float));

    }
}

Sqrt_Backward::~Sqrt_Backward() {
    if (!is_UVM) {

        CUDA_CALL(cudaFree(output_da));
        CUDA_CALL(cudaFree(input_a));
        CUDA_CALL(cudaFree(input_db));

    }
}

float Sqrt_Backward::Run() {
    if (is_UVM) {

        CUDA_CALL(cudaMallocManaged(&output_da, (long) N * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&input_a, (long) N * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&input_db, (long) N * sizeof(float)));

        CPUFillRand(input_a, (long) N * sizeof(float));
        CPUFillRand(input_db, (long) N * sizeof(float));

        GPUFillRand(output_da, (long) N * sizeof(float) * output_ratio);
        GPUFillRand(input_a, (long) N * sizeof(float) * input_ratio);
        GPUFillRand(input_db, (long) N * sizeof(float) * input_ratio);
        cudaDeviceSynchronize();
    }

    float milliseconds = 0;
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    
    CUDA_CALL(cudaEventRecord(start));
    sqrt_backward<<<batch_size, num_threads>>>(output_da, input_a, input_db, N);
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    
    return milliseconds;
}









Tanh_Forward::Tanh_Forward(cudnnHandle_t handle, vector<double> &args, bool is_UVM) : 
        handle(handle), is_UVM(is_UVM) {
    // 0. length     1. batch_size   2. num_threads
    N          = args[0];   batch_size = args[1];   num_threads= args[2];
    input_ratio = args[3]; output_ratio = args[4];
    // Alloc
    if (!is_UVM) {
        CUDA_CALL(cudaMalloc(&output, (long) N * sizeof(float)));
        CUDA_CALL(cudaMalloc(&input, (long) N * sizeof(float)));

        GPUFillRand(output, (long) N * sizeof(float));
        GPUFillRand(input, (long) N * sizeof(float));

    }
}

Tanh_Forward::~Tanh_Forward() {
    if (!is_UVM) {

        CUDA_CALL(cudaFree(output));
        CUDA_CALL(cudaFree(input));

    }
}

float Tanh_Forward::Run() {
    if (is_UVM) {

        CUDA_CALL(cudaMallocManaged(&output, (long) N * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&input, (long) N * sizeof(float)));

        CPUFillRand(input, (long) N * sizeof(float));

        GPUFillRand(output, (long) N * sizeof(float) * output_ratio);
        GPUFillRand(input, (long) N * sizeof(float) * input_ratio);
        cudaDeviceSynchronize();
    }

    float milliseconds = 0;
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    
    CUDA_CALL(cudaEventRecord(start));
    tanh_forward<<<batch_size, num_threads>>>(output, input, N);
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    
    return milliseconds;
}






Tanh_Backward::Tanh_Backward(cudnnHandle_t handle, vector<double> &args, bool is_UVM) : 
        handle(handle), is_UVM(is_UVM) {
    // 0. length     1. batch_size   2. num_threads
    N          = args[0];   batch_size = args[1];   num_threads= args[2];
    input_ratio = args[3]; output_ratio = args[4];
    // Alloc
    if (!is_UVM) {
        CUDA_CALL(cudaMalloc(&output_da, (long) N * sizeof(float)));
        CUDA_CALL(cudaMalloc(&input_a, (long) N * sizeof(float)));
        CUDA_CALL(cudaMalloc(&input_db, (long) N * sizeof(float)));

        GPUFillRand(output_da, (long) N * sizeof(float));
        GPUFillRand(input_a, (long) N * sizeof(float));
        GPUFillRand(input_db, (long) N * sizeof(float));

    }
}

Tanh_Backward::~Tanh_Backward() {
    if (!is_UVM) {

        CUDA_CALL(cudaFree(output_da));
        CUDA_CALL(cudaFree(input_a));
        CUDA_CALL(cudaFree(input_db));

    }
}

float Tanh_Backward::Run() {
    if (is_UVM) {

        CUDA_CALL(cudaMallocManaged(&output_da, (long) N * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&input_a, (long) N * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&input_db, (long) N * sizeof(float)));

        CPUFillRand(input_a, (long) N * sizeof(float));
        CPUFillRand(input_db, (long) N * sizeof(float));

        GPUFillRand(output_da, (long) N * sizeof(float) * output_ratio);
        GPUFillRand(input_a, (long) N * sizeof(float) * input_ratio);
        GPUFillRand(input_db, (long) N * sizeof(float) * input_ratio);
        cudaDeviceSynchronize();
    }

    float milliseconds = 0;
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    
    CUDA_CALL(cudaEventRecord(start));
    tanh_backward<<<batch_size, num_threads>>>(output_da, input_a, input_db, N);
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    
    return milliseconds;
}





Erf_Forward::Erf_Forward(cudnnHandle_t handle, vector<double> &args, bool is_UVM) : 
        handle(handle), is_UVM(is_UVM) {
    // 0. length     1. batch_size   2. num_threads
    N          = args[0];   batch_size = args[1];   num_threads= args[2];
    input_ratio = args[3]; output_ratio = args[4];
    // Alloc
    if (!is_UVM) {
        CUDA_CALL(cudaMalloc(&output, (long) N * sizeof(float)));
        CUDA_CALL(cudaMalloc(&input, (long) N * sizeof(float)));

        GPUFillRand(output, (long) N * sizeof(float));
        GPUFillRand(input, (long) N * sizeof(float));

    }
}

Erf_Forward::~Erf_Forward() {
    if (!is_UVM) {

        CUDA_CALL(cudaFree(output));
        CUDA_CALL(cudaFree(input));

    }
}

float Erf_Forward::Run() {
    if (is_UVM) {

        CUDA_CALL(cudaMallocManaged(&output, (long) N * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&input, (long) N * sizeof(float)));

        CPUFillRand(input, (long) N * sizeof(float));

        GPUFillRand(output, (long) N * sizeof(float) * output_ratio);
        GPUFillRand(input, (long) N * sizeof(float) * input_ratio);
        cudaDeviceSynchronize();
    }

    float milliseconds = 0;
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    
    CUDA_CALL(cudaEventRecord(start));
    erf_forward<<<batch_size, num_threads>>>(output, input, N);
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    
    return milliseconds;
}








Erf_Backward::Erf_Backward(cudnnHandle_t handle, vector<double> &args, bool is_UVM) : 
        handle(handle), is_UVM(is_UVM) {
    // 0. length     1. batch_size   2. num_threads
    N          = args[0];   batch_size = args[1];   num_threads= args[2];
    input_ratio = args[3]; output_ratio = args[4];
    // Alloc
    if (!is_UVM) {
        CUDA_CALL(cudaMalloc(&output_da, (long) N * sizeof(float)));
        CUDA_CALL(cudaMalloc(&input_a, (long) N * sizeof(float)));
        CUDA_CALL(cudaMalloc(&input_db, (long) N * sizeof(float)));

        GPUFillRand(output_da, (long) N * sizeof(float));
        GPUFillRand(input_a, (long) N * sizeof(float));
        GPUFillRand(input_db, (long) N * sizeof(float));

    }
}

Erf_Backward::~Erf_Backward() {
    if (!is_UVM) {

        CUDA_CALL(cudaFree(output_da));
        CUDA_CALL(cudaFree(input_a));
        CUDA_CALL(cudaFree(input_db));

    }
}

float Erf_Backward::Run() {
    if (is_UVM) {

        CUDA_CALL(cudaMallocManaged(&output_da, (long) N * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&input_a, (long) N * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&input_db, (long) N * sizeof(float)));

        CPUFillRand(input_a, (long) N * sizeof(float));
        CPUFillRand(input_db, (long) N * sizeof(float));

        GPUFillRand(output_da, (long) N * sizeof(float) * output_ratio);
        GPUFillRand(input_a, (long) N * sizeof(float) * input_ratio);
        GPUFillRand(input_db, (long) N * sizeof(float) * input_ratio);
        cudaDeviceSynchronize();
    }

    float milliseconds = 0;
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    
    CUDA_CALL(cudaEventRecord(start));
    erf_backward<<<batch_size, num_threads>>>(output_da, input_a, input_db, N);
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    
    return milliseconds;
}










Sum_Forward::Sum_Forward(cudnnHandle_t handle, vector<double> &args, bool is_UVM) : 
        handle(handle), is_UVM(is_UVM) {
    // 0. length     1. batch_size   2. num_threads   3. dim_3
    N          = args[0];   batch_size = args[1];   num_threads= args[2];
    dim_3      = args[3];   input_ratio = args[4]; output_ratio = args[5];
    // Alloc
    if (!is_UVM) {
        CUDA_CALL(cudaMalloc(&output_b, (long) N * sizeof(float)));
        CUDA_CALL(cudaMalloc(&input_a, (long) N * dim_3 * sizeof(float)));

        GPUFillRand(output_b, (long) N * sizeof(float));
        GPUFillRand(input_a, (long) N * dim_3 * sizeof(float));

    }
}

Sum_Forward::~Sum_Forward() {
    if (!is_UVM) {

        CUDA_CALL(cudaFree(output_b));
        CUDA_CALL(cudaFree(input_a));
    }
}

float Sum_Forward::Run() {
    if (is_UVM) {

        CUDA_CALL(cudaMallocManaged(&output_b, (long) N * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&input_a, (long) N * dim_3 * sizeof(float)));

        CPUFillRand(input_a, (long) N * dim_3 * sizeof(float));
        CPUFillRand(output_b, (long) N * sizeof(float));

        GPUFillRand(output_b, (long) N * sizeof(float) * output_ratio);
        GPUFillRand(input_a, (long) N * dim_3 * sizeof(float) * input_ratio);
        cudaDeviceSynchronize();
    }

    float milliseconds = 0;
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    
    CUDA_CALL(cudaEventRecord(start));
    sum_forward<<<batch_size, num_threads>>>(input_a, output_b, dim_3, N);
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    
    return milliseconds;
}





Sum_Backward::Sum_Backward(cudnnHandle_t handle, vector<double> &args, bool is_UVM) : 
        handle(handle), is_UVM(is_UVM) {
    // 0. length     1. batch_size   2. num_threads   3. dim_3
    N          = args[0];   batch_size = args[1];   num_threads= args[2];
    dim_3      = args[3];   input_ratio = args[4]; output_ratio = args[5];
    // Alloc
    if (!is_UVM) {
        CUDA_CALL(cudaMalloc(&output_da, (long) N * dim_3 * sizeof(float)));
        CUDA_CALL(cudaMalloc(&input_db, (long) N * sizeof(float)));

        GPUFillRand(output_da, (long) N * sizeof(float));
        GPUFillRand(input_db, (long) N * sizeof(float));

    }
}

Sum_Backward::~Sum_Backward() {
    if (!is_UVM) {

        CUDA_CALL(cudaFree(output_da));
        CUDA_CALL(cudaFree(input_db));
    }
}

float Sum_Backward::Run() {
    if (is_UVM) {

        CUDA_CALL(cudaMallocManaged(&output_da, (long) N * dim_3 * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&input_db, (long) N * sizeof(float)));

        CPUFillRand(input_db, (long)N * sizeof(float));

        GPUFillRand(output_da, (long) N * dim_3 * sizeof(float) * output_ratio);
        GPUFillRand(input_db, (long) N * sizeof(float) * input_ratio);
        cudaDeviceSynchronize();
    }

    float milliseconds = 0;
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    
    CUDA_CALL(cudaEventRecord(start));
    sum_backward<<<batch_size, num_threads>>>(output_da, input_db, dim_3, N);
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    
    return milliseconds;
}




GatherV2_Forward::GatherV2_Forward(cudnnHandle_t handle, vector<double> &args, bool is_UVM) : 
        handle(handle), is_UVM(is_UVM) {
    // 0. length     1. batch_size   2. num_threads
    N          = args[0];   batch_size = args[1];   num_threads= args[2];
    input_ratio = args[3]; output_ratio = args[4];
    // Alloc
    if (!is_UVM) {
        CUDA_CALL(cudaMalloc(&output, (long) N * sizeof(float)));
        CUDA_CALL(cudaMalloc(&input, (long) N * sizeof(float)));

        GPUFillRand(output, (long) N * sizeof(float));
        GPUFillRand(input, (long) N * sizeof(float));

    }
}

GatherV2_Forward::~GatherV2_Forward() {
    if (!is_UVM) {

        CUDA_CALL(cudaFree(output));
        CUDA_CALL(cudaFree(input));

    }
}

float GatherV2_Forward::Run() {
    if (is_UVM) {

        CUDA_CALL(cudaMallocManaged(&output, (long) N * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&input, (long) N * sizeof(float)));

        CPUFillRand(input, (long) N * sizeof(float));

        GPUFillRand(output, (long) N * sizeof(float) * output_ratio);
        GPUFillRand(input, (long) N * sizeof(float) * input_ratio);
        cudaDeviceSynchronize();
    }

    float milliseconds = 0;
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    
    CUDA_CALL(cudaEventRecord(start));
    gather<<<batch_size, num_threads>>>(output, input, N);
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    
    return milliseconds;
}



GatherV2_Backward::GatherV2_Backward(cudnnHandle_t handle, vector<double> &args, bool is_UVM) : 
        handle(handle), is_UVM(is_UVM) {
    // 0. length     1. batch_size   2. num_threads
    N          = args[0];   batch_size = args[1];   num_threads= args[2];
    input_ratio = args[3]; output_ratio = args[4];
    // Alloc
    if (!is_UVM) {
        CUDA_CALL(cudaMalloc(&output_da, (long) N * sizeof(float)));

        CUDA_CALL(cudaMalloc(&input_db, (long) N * sizeof(float)));

        GPUFillRand(output_da, (long) N * sizeof(float));

        GPUFillRand(input_db, (long) N * sizeof(float));

    }
}

GatherV2_Backward::~GatherV2_Backward() {
    if (!is_UVM) {

        CUDA_CALL(cudaFree(output_da));

        CUDA_CALL(cudaFree(input_db));

    }
}

float GatherV2_Backward::Run() {
    if (is_UVM) {

        CUDA_CALL(cudaMallocManaged(&output_da, (long) N * sizeof(float)));

        CUDA_CALL(cudaMallocManaged(&input_db, (long) N * sizeof(float)));

        CPUFillRand(input_db, (long) N * sizeof(float));

        GPUFillRand(output_da, (long) N * sizeof(float) * output_ratio);

        GPUFillRand(input_db, (long) N * sizeof(float) * input_ratio);
        cudaDeviceSynchronize();
    }

    float milliseconds = 0;
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    
    CUDA_CALL(cudaEventRecord(start));
    gather<<<batch_size, num_threads>>>(output_da, input_db, N);
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    
    return milliseconds;
}







ApplyGrad::ApplyGrad(cudnnHandle_t handle, vector<double> &args, bool is_UVM) : 
        handle(handle), is_UVM(is_UVM) {
    // 0. length     1. batch_size   2. num_threads
    N          = args[0];   batch_size = args[1];   num_threads= args[2];
    input_ratio = args[3]; output_ratio = args[4];
    // Alloc
    if (!is_UVM) {
        CUDA_CALL(cudaMalloc(&output_data, (long) N * sizeof(float)));
        CUDA_CALL(cudaMalloc(&d_output_data, (long) N * sizeof(float)));
        GPUFillRand(output_data, (long) N * sizeof(float));
        GPUFillRand(d_output_data, (long) N * sizeof(float));
    }
}

ApplyGrad::~ApplyGrad() {
    if (!is_UVM) {
        CUDA_CALL(cudaFree(output_data));
        CUDA_CALL(cudaFree(d_output_data));
    }
}

float ApplyGrad::Run() {
    if (is_UVM) {
        CUDA_CALL(cudaMallocManaged(&output_data, (long) N * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&d_output_data, (long) N * sizeof(float)));
        CPUFillRand(output_data, (long) N * sizeof(float));
        CPUFillRand(d_output_data, (long) N * sizeof(float));

        GPUFillRand(output_data, (long) N * sizeof(float) * output_ratio);
        GPUFillRand(d_output_data, (long) N * sizeof(float) * input_ratio);
        cudaDeviceSynchronize();
    }

    float milliseconds = 0;
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    
    CUDA_CALL(cudaEventRecord(start));
    apply_grad<<<batch_size, num_threads>>>(output_data, d_output_data, N);
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    
    if (is_UVM) {
        CUDA_CALL(cudaFree(output_data));
        CUDA_CALL(cudaFree(d_output_data));
    }
    return milliseconds;
}



MakeError::MakeError(cudnnHandle_t handle, vector<double> &args, bool is_UVM) : 
        handle(handle), is_UVM(is_UVM) {
    // 0. length     1. batch_size   2. num_threads
    N          = args[0];   batch_size = args[1];   num_threads= args[2];
    input_ratio = args[3]; output_ratio = args[4];
    // Alloc
    if (!is_UVM) {
        CUDA_CALL(cudaMalloc(&output_data, (long) N * sizeof(float)));
        CUDA_CALL(cudaMalloc(&error_data, (long) N * sizeof(float)));
        GPUFillRand(output_data, (long) N * sizeof(float));
    }
}

MakeError::~MakeError() {
    if (!is_UVM) {
        CUDA_CALL(cudaFree(output_data));
        CUDA_CALL(cudaFree(error_data));
    }
}

float MakeError::Run() {
    if (is_UVM) {
        CUDA_CALL(cudaMallocManaged(&output_data, (long) N * sizeof(float)));
        CUDA_CALL(cudaMallocManaged(&error_data, (long) N * sizeof(float)));
        CPUFillRand(output_data, (long) N * sizeof(float));

        GPUFillRand(output_data, (long) N * sizeof(float) * input_ratio);
        GPUFillRand(error_data, (long) N * sizeof(float) * output_ratio);
        cudaDeviceSynchronize();
    }

    float milliseconds = 0;
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    
    CUDA_CALL(cudaEventRecord(start));
    makeError<<<batch_size, num_threads>>>(error_data, output_data, 1, N);
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    
    if (is_UVM) {
        CUDA_CALL(cudaFree(output_data));
        CUDA_CALL(cudaFree(error_data));
    }
    return milliseconds;
}



TransferA0::TransferA0(cudnnHandle_t handle, vector<double> &args, bool is_UVM) : 
        handle(handle), is_UVM(is_UVM) {
    // 0. length     1. batch_size   2. num_threads
    N          = args[0];   batch_size = args[1];   num_threads= args[2];
    // Alloc
    data = (float *) malloc((long) N * sizeof(float));
    if (!is_UVM) {
        CUDA_CALL(cudaMalloc(&activation_tensor, (long) N * sizeof(float)));
    }
}

TransferA0::~TransferA0() {
    free(data);
    if (!is_UVM) {
        CUDA_CALL(cudaFree(activation_tensor));
    }
}

float TransferA0::Run() {
    if (is_UVM) {
        CUDA_CALL(cudaMallocManaged(&activation_tensor, (long) N * sizeof(float)));
        cudaDeviceSynchronize();
    }

    float milliseconds = 0;
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    
    CUDA_CALL(cudaEventRecord(start));
    cudaMemcpy(activation_tensor, data, N * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    
    if (is_UVM) {
        CUDA_CALL(cudaFree(activation_tensor));
    }
    return milliseconds;
}
