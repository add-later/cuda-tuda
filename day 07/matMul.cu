%%cuda
#include <iostream>
#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K){
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;

    float sum = 0.0;
    if (row < M && col < K) {
    for (int i=0; i<N; i++){
            sum += A[row * N + i] * B[i * K + col];
            }
    C[row * K + col] = sum;
    }
}

int main(){
    using namespace std;
    int N = 2;

    size_t size = N * N * sizeof(float);

    float *h_a = (float*)malloc(size);
    float * h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    for (int i = 0; i < N * N; i++){
        h_a[i] = i + 1.0f;
        h_b[i] = i + 5.0f;
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (N + threadsPerBlock.x -1 ) / threadsPerBlock.x,
        (N + threadsPerBlock.y - 1) / threadsPerBlock.y
    );
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N, N, N);

    cudaDeviceSynchronize();
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < N * N; i++){
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