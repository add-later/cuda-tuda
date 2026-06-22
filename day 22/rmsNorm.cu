%%cuda
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <iostream>

__global__ void computeSum(const float* input, float* sumOfSquares, int N) {
    __shared__ float temp[256];
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    temp[threadIdx.x] = (idx < N) ? input[idx] * input[idx] : 0;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride)
            temp[threadIdx.x] += temp[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicAdd(sumOfSquares, temp[0]);
}

__global__ void normalize(const float* input, float* sumOfSquares,
                          float gamma, float beta,
                          float* output, int N, float eps) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) {
        float rms = sqrtf(sumOfSquares[0] / N + eps);
        output[idx] = gamma * (input[idx] / rms) + beta;
    }
}

// ── solve ──────────────────────────────────────────────────────────────────
extern "C" void solve(const float* input, float gamma, float beta,
                      float* output, int N, float eps) {
    int threadsPerBlock = 256;
    int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

    float* sumOfSquares;
    cudaMalloc(&sumOfSquares, sizeof(float));
    cudaMemset(sumOfSquares, 0, sizeof(float)); 

    computeSum<<<blocksPerGrid, threadsPerBlock>>>(input, sumOfSquares, N);
    cudaDeviceSynchronize();
    normalize<<<blocksPerGrid, threadsPerBlock>>>(input, sumOfSquares,
                                                  gamma, beta, output, N, eps);
    cudaDeviceSynchronize();
    cudaFree(sumOfSquares);
}

int main() {
    using namespace std;
    const int N      = 3;
    float gamma      = 1.0f;
    float beta       = 0.0f;
    float eps        = 1e-5f;
    float h_in[N]    = {1.0f, 2.0f, 3.0f};
    float h_out[N]   = {};

    float *d_in, *d_out;
    cudaMalloc(&d_in,  N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    solve(d_in, gamma, beta, d_out, N, eps);

    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);

    for (int i = 0; i < N; i++){
        cout<<h_out[i]<<" ";
        
    }
    return 0;
}
