#include <iostream>
#include <cuda_runtime.h>
#include <math.h>

using namespace std;

__global__ void geglu_kernel(const float* input, float* output, int halfN) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < halfN) {
        output[idx] = input[idx] * 0.5 * input[idx + halfN] * (1 + erf(input[idx + halfN] * 0.70710678f));
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    geglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}


int main() {
    const int N = 2;

    float A[N] = {1.0f, 1.0f};
    float output[N];

    float *d_A, *d_output;

    cudaMalloc((void**)&d_A, N * sizeof(float));
    cudaMalloc((void**)&d_output, N * sizeof(float));

    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);

    solve(d_A, d_output, N);

    cudaMemcpy(output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    cout << "Output: [";
    for (int i = 0; i < N/2; i++) {
        cout << output[i];
    }
    cout << "]" << endl;

    cudaFree(d_A);
    cudaFree(d_output);

    return 0;
}