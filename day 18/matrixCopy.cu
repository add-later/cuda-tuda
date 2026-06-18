#include <iostream>
#include <cuda_runtime.h>

__global__ void copy_matrix_kernel(const float* A, float* B, int total) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < total){
        B[idx] = A[idx];
    }
}

int main(){
    using namespace std;
    int N = 2;
    int total = N * N;

    size_t size = N * N * sizeof(float);

    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);

    for (int i = 0; i < N * N; i++){
        h_a[i] = i + 1.0f;
    }

    float *d_a, *d_b;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;
    copy_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, total);

    cudaDeviceSynchronize();
    cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N * N; i++){
        cout << h_b[i] << " ";
        if ((i + 1) % N == 0) {
            cout << endl;
        }
    }

    cudaFree(d_a);
    cudaFree(d_b);
    free(h_a);
    free(h_b);

    return 0;
}