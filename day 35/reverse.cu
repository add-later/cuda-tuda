#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__global__ void reverse_array(float* input, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N / 2) {
        float temp = input[N - 1 - idx];
        input[N - 1 - idx] = input[idx];
        input[idx] = temp; 
    }
}

extern "C" void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; 

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}

int main() {
    const int N = 4;
    float A[N] = {1.0f, 2.0f, 3.0f, 4.0f};
    float *d_A;
    
    cudaMalloc((void**)&d_A, N * sizeof(float));
    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
    solve(d_A, N);
    cudaMemcpy(A, d_A, N * sizeof(float), cudaMemcpyDeviceToHost);

    cout << "Output: [ ";
    for (int i = 0; i < N; i++) {
        cout << A[i] << " ";
    }
    cout << "]" << endl;
    
    cudaFree(d_A);

    return 0;
}