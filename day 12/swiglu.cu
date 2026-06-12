#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
__global__ void swiglu_kernel(const float* input, float* output, int halfN) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < halfN){
    output[idx] = input[idx + halfN] * 1/(1 + expf(-input[idx])) * input[idx];
    }
}

int main() {
    using namespace std;
    int N = 4;
    float h_input[4] = {1.0, 2.0, 3.0, 4.0};
    float *h_output = new float[N];
    int halfN = N / 2;

    float *d_input, *d_output;

    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, halfN);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < halfN; ++i) {
        cout << h_output[i] << " ";
    }

    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_output;

    return 0;
}