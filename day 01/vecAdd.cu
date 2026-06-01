#include <iostream>
#include <cuda_runtime.h>

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    using namespace std;

    const int N = 10;
    size_t size = N * sizeof(float);
    
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, N);

    cudaDeviceSynchronize();
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a); 
    cudaFree(d_b); 
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}