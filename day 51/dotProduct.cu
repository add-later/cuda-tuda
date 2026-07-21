#include <cuda_runtime.h>
#include <iostream>
using namespace std;

__global__ void dot_product(const float* A, const float* B, float* result, int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float val = (idx < N) ? A[idx] * B[idx] : 0.0f;
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(result, sdata[0]);
}

extern "C" void solve(const float* A, const float* B, float* result, int N) {
    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    cudaMemset(result, 0, sizeof(float));
    dot_product<<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(float)>>>(A, B, result, N);
    cudaDeviceSynchronize();
}

int main() {
    const int N = 4;

    float h_A[N] = {1.0f, 2.0f, 3.0f, 4.0f};
    float h_B[N] = {5.0f, 6.0f, 7.0f, 8.0f};
    float h_result = 0.0f;

    float *d_A, *d_B, *d_result;

    cudaMalloc((void**)&d_A, N * sizeof(float));
    cudaMalloc((void**)&d_B, N * sizeof(float));
    cudaMalloc((void**)&d_result, sizeof(float));

    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    solve(d_A, d_B, d_result, N);

    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    cout << "result = " << h_result << endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_result);

    return 0;
}