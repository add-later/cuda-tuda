#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* input, float* output, int N) {
    __shared__ float sdata[256];
    int tid = threadIdx.x; 
    int row = blockIdx.x;
    const float* x = input + row * N;
    float* y = output + row * N;

    float local_max = -INFINITY;
    for (int i = tid; i < N; i += blockDim.x) {
        local_max = fmaxf(local_max, x[i]);
    }
    sdata[tid] = local_max;
    __syncthreads();

    for (int i = blockDim.x / 2; i >= 1; i /= 2) {
        if (tid < i) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + i]);
        }
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x){
     local_sum += expf(x[i] - row_max);   
    }
    
    sdata[tid]= local_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float row_sum = sdata[0];
    __syncthreads();

    for (int i = tid; i < N; i += blockDim.x){
        y[i] = expf(x[i] - row_max) / row_sum;
    }
    
}
bool is_power_of_two(int x) {
    return x > 0 && (x & (x - 1)) == 0;
}

int main() {
    using namespace std;
    int N = 3;
    float h_input[3] = {1.0f, 2.0f, 3.0f};
    float *h_output = new float[N];
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    if (!is_power_of_two(threads_per_block)) {
        std::cerr << "threads_per_block must be a power of two\n";
        return 1;
    }
    int blocks_per_grid = 1; // one block per row; we have 1 row here

    softmax_kernel<<<blocks_per_grid, threads_per_block>>>(d_input, d_output, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << "\n";
        return 1;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; ++i) {
        cout << h_output[i] << " ";
    }
    cout << "\n";
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_output;
    return 0;
}