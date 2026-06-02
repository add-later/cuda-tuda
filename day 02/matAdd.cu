#include <iostream>
#include <cuda_runtime.h>

__global__ void matrix_add(const float* A, const float* B, float* C, int N) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < N * N){
        C[row] = A[row] + B[row];
    }
}

int main() {
    using namespace std;

    const int N = 5;
    size_t size = N * N * sizeof(float);
    
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    for (int i = 0; i < N * N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid  = (N * N + threadsPerBlock - 1) / threadsPerBlock;
    matrix_add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    cudaDeviceSynchronize();
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < N * N; i++) {
        cout << h_c[i] << " ";
        if ((i + 1) % N == 0) {
            cout << endl;
        }
    }
    cudaFree(d_a); 
    cudaFree(d_b); 
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}