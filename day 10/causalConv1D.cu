#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

__global__ void solve(const float* x, const float* weight, const float* bias, float* output, int B,
                      int L, int D, int K) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int b = i / (L * D);
    int remainder = i % (L * D);
    int l = remainder / D;
    int d = remainder % D;
    
    if (i < B * L * D) {
        float sum = 0.0;
        for (int k = 0; k < K; k++) {
            if (l - k >= 0) {
                sum += weight[d * K + k] * x[b * L * D + (l - k) * D + d];
            }
        }
        output[b * L * D + l * D + d] = sum + bias[d];
    }
}

int main() {
    using namespace std;
    int B = 1, L = 4, D = 2, K = 3;
    
    vector<float> h_x = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f
    };
    
    vector<float> h_weight = {
        1.0f, 0.0f, -1.0f,
        1.0f, 1.0f, 1.0f
    };
    
    vector<float> h_bias = {0.0f, 0.0f};
    vector<float> h_output(B * L * D, 0.0f);
    
    float *d_x, *d_weight, *d_bias, *d_output;
    
    cudaMalloc(&d_x, h_x.size() * sizeof(float));
    cudaMalloc(&d_weight, h_weight.size() * sizeof(float));
    cudaMalloc(&d_bias, h_bias.size() * sizeof(float));
    cudaMalloc(&d_output, h_output.size() * sizeof(float));
    
    cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight.data(), h_weight.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias.data(), h_bias.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    int total_threads = B * L * D;
    int threads_per_block = 256;
    int blocks_per_grid = (total_threads + threads_per_block - 1) / threads_per_block;
    
    solve<<<blocks_per_grid, threads_per_block>>>(d_x, d_weight, d_bias, d_output, B, L, D, K);
    
    cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    for (int b = 0; b < B; ++b) {
        for (int l = 0; l < L; ++l) {
            for (int d = 0; d < D; ++d) {
                cout << h_output[b * L * D + l * D + d] << " ";
            }
            cout << "\n";
        }
    }
    
    cudaFree(d_x);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_output);
    
    return 0;
}