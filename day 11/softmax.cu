%%cuda
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* input, float* output, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) {
        float max_val = input[0];
        for (int i = 1; i < N; i++) {
            if (input[i] > max_val) {
                max_val = input[i];
            }
        }
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += expf(input[i] - max_val);
        }
        output[idx] = expf(input[idx] - max_val) / sum;
    }
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
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    softmax_kernel<<<blocks_per_grid, threads_per_block>>>(d_input, d_output, N);

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