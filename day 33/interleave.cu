#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void interleave_kernel(const float* A, const float* B, float* output, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < 2*N){
        if (idx % 2 == 0){
            output[idx] = A[idx/2];
        }
        else {
            output[idx] = B[idx/2];
        }
    }
}

extern "C" void solve(const float* A, const float* B, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (2*N + threadsPerBlock - 1) / threadsPerBlock;

    interleave_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, output, N);
    cudaDeviceSynchronize();
}

int main() {
    const int N = 3;

    float A[N] = {1.0f, 2.0f, 3.0f};
    float B[N] = {4.0f, 5.0f, 6.0f};
    float output[2 * N];

    float *d_A, *d_B, *d_output;

    cudaMalloc((void**)&d_A, N * sizeof(float));
    cudaMalloc((void**)&d_B, N * sizeof(float));
    cudaMalloc((void**)&d_output, 2 * N * sizeof(float));

    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);

    solve(d_A, d_B, d_output, N);

    cudaMemcpy(output, d_output, 2 * N * sizeof(float), cudaMemcpyDeviceToHost);

    cout << "Output: [";
    for (int i = 0; i < 2 * N; i++) {
        cout << output[i];
        if (i != 2 * N - 1) cout << ", ";
    }
    cout << "]" << endl;
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_output);

    return 0;
}