#include <cuda_runtime.h>
#include <iostream>
using namespace std;

// Q, cos, sin, output are device pointers
__global__ void rope_kernel(float* Q, float* cos, float* sin, float* output, int M, int D){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // rotation
    if (idx < M * D){
        // reading index, finding out which is row, which is col
        int row = idx / D;
        int col = idx % D;
        // half rotate
        int half = D / 2;
        float temp = 0.0f;
        if (col < half) {
            int half_col = row * D + col + half;
            temp = Q[half_col] * (-1.0f);
        }
        else {
            int half_col = row * D + col - half;
            temp = Q[half_col];
        }
        output[idx] = Q[idx] * cos[idx] + temp * sin[idx];
    }
}


int main(){
    const int M = 2;
    const int D = 4;
    const int N = M * D;

    float Q_host[M][D] = {
        {1.0f, 2.0f, 3.0f, 4.0f},
        {1.0f, 1.0f, 1.0f, 1.0f}
    };
    float Cos_host[M][D] = {
        {1.0f, 1.0f, 1.0f, 1.0f},
        {0.0f, 0.0f, 0.0f, 0.0f}
    };
    float Sin_host[M][D] = {
        {0.0f, 0.0f, 0.0f, 0.0f},
        {1.0f, 1.0f, 1.0f, 1.0f}
    };
    float out_host[M][D] = {};  // zero-initialized

    float *d_Q, *d_Cos, *d_Sin, *d_out;
    cudaMalloc(&d_Q,   N * sizeof(float));
    cudaMalloc(&d_Cos, N * sizeof(float));
    cudaMalloc(&d_Sin, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));

    cudaMemcpy(d_Q,   &Q_host[0][0],   N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Cos, &Cos_host[0][0], N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sin, &Sin_host[0][0], N * sizeof(float), cudaMemcpyHostToDevice);

    // run the kernel
    int threadsPerBlock = 256;
    int output_size = M * D;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;
    rope_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_Q, d_Cos, d_Sin, d_out, M, D);

    cudaMemcpy(&out_host[0][0], d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_Cos);
    cudaFree(d_Sin);
    cudaFree(d_out);
               
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < D; ++j) {
            cout << out_host[i][j] << (j == D - 1 ? "" : ", ");
        }
        cout << (i == M - 1 ? "" : ",\n");
    }

    return 0;
}
