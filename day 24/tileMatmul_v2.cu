#include <iostream>
#include <cuda_runtime.h>
#define TILE_WIDTH 16

__global__ void tiled_matmul_kernel(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.x;
    int col = blockIdx.x * TILE_WIDTH;
    int ty = threadIdx.x;

    float sum[TILE_WIDTH] = {0.0f};

    for (int m = 0; m < (N + TILE_WIDTH - 1) / TILE_WIDTH; m++) {
        // READING
        for (int i = 0; i < TILE_WIDTH; i++) {
            if (row < N && (m * TILE_WIDTH + i) < N)
                As[ty][i] = A[row * N + (m * TILE_WIDTH + i)];
            else
                As[ty][i] = 0.0f;

            if ((m * TILE_WIDTH + i) < N && (col + ty) < N){
                Bs[i][ty] = B[(m * TILE_WIDTH + i) * N + col + ty];
            }
            else
                Bs[i][ty] = 0.0f;
        }
        __syncthreads();

        // CALCULATION

        for (int i = 0; i < TILE_WIDTH; i++) {
            for (int k = 0; k < TILE_WIDTH; k++) {
                sum[i] += As[ty][k] * Bs[k][i];
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < TILE_WIDTH; i++) {
        if (row < N && (col + i) < N)
            C[row * N + col + i] = sum[i];
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

    int threadsPerBlock = TILE_WIDTH;
    dim3 blocksPerGrid(
    (N + TILE_WIDTH - 1) / TILE_WIDTH,
    (N + TILE_WIDTH - 1) / TILE_WIDTH
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