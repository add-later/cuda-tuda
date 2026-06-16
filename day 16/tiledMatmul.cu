#include <iostream>
#include <cuda_runtime.h>
#define TILE_WIDTH 16

__global__ void tiled_matmul_kernel(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    float sum = 0.0;

    for (int m = 0; m < (N + TILE_WIDTH - 1) / TILE_WIDTH; m++) {
        // READING
        if (row < N && (m * TILE_WIDTH + tx) < N) {
            As[ty][tx] = A[row * N + (m * TILE_WIDTH + tx)];
        } else {
            As[ty][tx] = 0.0;
        }

        if (col < N && (m * TILE_WIDTH + ty) < N) {
            Bs[ty][tx] = B[(m * TILE_WIDTH + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0;
        }
        __syncthreads();

        // CALCULATION

        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

int main(){
    using namespace std;
    int N = 2;

    size_t size = N * N * sizeof(float);

    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
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

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH); // Match your tile width
    dim3 blocksPerGrid(
        (N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (N + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    tiled_matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

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