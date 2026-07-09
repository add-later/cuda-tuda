#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#define MAX_D 128

__global__ void attn_kernel(const float* Q, const float* K, const float* V, 
                            float* output, int M, int N, int d){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    const float scale = 1.0f / sqrtf((float)d);
    const float* q = Q + row * d;

    float maxScore = -INFINITY;
    for (int i = 0; i< N; i++){
        const float* k = K + i * d;
        float dot = 0.0f;
        for (int j = 0; j < d; j++){
            dot += q[j] * k[j];
        }
        dot *= scale;
        maxScore = fmaxf(maxScore, dot);
    }

    float sumExp = 0.0f;
    float acc[MAX_D];
    for (int i = 0; i < d; i++) {
        acc[i] = 0.0f;
    }

    for (int i = 0; i < N; i++){
        const float* k = K + i * d;
        float dot = 0.0f;
        for (int j = 0; j < d; j++){
            dot += q[j] * k[j];
        }
        dot *= scale;

        float w = expf(dot - maxScore);
        sumExp += w;
        const float* v = V + i * d;
        for (int t = 0; t < d; t++){
            acc[t] += w * v[t];
        }
    }

    float* out = output + row * d;
    for (int i = 0; i < d; i++){
        out[i] = acc[i] / sumExp;
    }

                            }
void solve(const float* Q, const float* K, const float* V, float* output, int M, int N,
                      int d) {
    int threadsPerBlock = 256;
    int blocks = (M + threadsPerBlock - 1) / threadsPerBlock;

    attn_kernel<<<blocks, threadsPerBlock>>>(Q, K, V, output, M, N, d);
    cudaDeviceSynchronize();
                      }


int main() {
    const int M = 2, N = 3, d = 4;

    float h_Q[M * d] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f
    };
    float h_K[N * d] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f
    };
    float h_V[N * d] = {
        1.0f,  2.0f,  3.0f,  4.0f,
        5.0f,  6.0f,  7.0f,  8.0f,
        9.0f, 10.0f, 11.0f, 12.0f
    };
    float h_output[M * d] = {0};

    float *d_Q, *d_K, *d_V, *d_output;
    cudaMalloc(&d_Q, M * d * sizeof(float));
    cudaMalloc(&d_K, N * d * sizeof(float));
    cudaMalloc(&d_V, N * d * sizeof(float));
    cudaMalloc(&d_output, M * d * sizeof(float));

    cudaMemcpy(d_Q, h_Q, M * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, N * d * sizeof(float), cudaMemcpyHostToDevice);

    solve(d_Q, d_K, d_V, d_output, M, N, d);

    cudaMemcpy(h_output, d_output, M * d * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < d; j++) {
            printf("%8.4f ", h_output[i * d + j]);
        }
        printf("\n");
    }

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);

    return 0;
}