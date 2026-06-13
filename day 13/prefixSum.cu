#include <cuda_runtime.h>
#include <iostream>
__global__ void prefix_sum_kernel(const float* input, float *output, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N){
        float sum = input[idx];
        for (int i=0; i < idx; i++) {
            sum += input[i];
        }
        output[idx] = sum;
    }
}

int main() {
    using namespace std;

    int N = 4;
    float h_input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float *h_output = new float[N];

    float *d_input, *d_output;

    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    prefix_sum_kernel<<<blocks_per_grid, threads_per_block>>>(d_input, d_output, N);

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