#include <cuda_runtime.h>
#include <iostream>
__global__ void histogram(const int* input, int* histogram, int N, int num_bins) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) {
        int bin = input[idx];
        atomicAdd(histogram + bin, 1);
    }
}

int main() {
    using namespace std;

    int N = 5;
    int num_bins = 3;
    int h_input[5] = {0, 1, 2, 1, 0};
    int *h_output = new int[num_bins];

    int *d_input, *d_output;

    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    histogram<<<blocks_per_grid, threads_per_block>>>(d_input, d_output, N, num_bins);

    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_bins; ++i) {
        cout << h_output[i] << " ";
    }
    cout << "\n";

    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_output;

    return 0;
}