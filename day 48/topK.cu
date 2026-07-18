#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <vector>

__global__ void top_k(const float* __restrict__ input,
                       int* __restrict__ indices,
                       float* __restrict__ values,
                       int N, int K,
                       bool* __restrict__ selected_buf) {
                           
    int row = blockIdx.x;

    const float* row_input = input + (size_t)row * N;
    int* row_indices = indices + (size_t)row * K;
    float* row_values = values + (size_t)row * K;
    bool* selected = selected_buf + (size_t)row * N;

    if (threadIdx.x == 0) {
        for (int i = 0; i < N; i++) {
            selected[i] = false;
        }

        for (int k = 0; k < K; k++) {
            float max_val = -INFINITY;
            int max_idx = -1;

            for (int i = 0; i < N; i++) {
                if (!selected[i] && row_input[i] > max_val) {
                    max_val = row_input[i];
                    max_idx = i;
                }
            }

            if (max_idx >= 0) {
                selected[max_idx] = true;
                row_values[k] = max_val;
                row_indices[k] = max_idx;
            } else {
                row_values[k] = -INFINITY;
                row_indices[k] = -1;
            }
        }

        float max_for_stability = -INFINITY;
        for (int k = 0; k < K; k++) {
            if (row_values[k] > max_for_stability) {
                max_for_stability = row_values[k];
            }
        }

        float sum_exp = 0.0f;
        for (int k = 0; k < K; k++) {
            float e = expf(row_values[k] - max_for_stability);
            row_values[k] = e;
            sum_exp += e;
        }

        for (int k = 0; k < K; k++) {
            row_values[k] = row_values[k] / sum_exp;
        }
    }
}

void solve(const float* logits, float* topk_weights, int* topk_indices, int M, int E,
                       int k) {
    dim3 block_size(1);
    dim3 grid_size(M);

    bool* selected;
    cudaMalloc(&selected, (size_t)M * E * sizeof(bool));

    top_k<<<grid_size, block_size>>>(logits, topk_indices, topk_weights, E, k, selected);
    cudaDeviceSynchronize();

    cudaFree(selected);
}

int main() {
    const int M = 2;
    const int E = 4;
    const int K = 2;

    float h_logits[M * E] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        4.0f, 3.0f, 2.0f, 1.0f
    };

    float h_topk_weights[M * K] = {0};
    int h_topk_indices[M * K] = {0};

    float* d_logits;
    float* d_topk_weights;
    int* d_topk_indices;

    cudaMalloc(&d_logits, M * E * sizeof(float));
    cudaMalloc(&d_topk_weights, M * K * sizeof(float));
    cudaMalloc(&d_topk_indices, M * K * sizeof(int));

    cudaMemcpy(d_logits, h_logits, M * E * sizeof(float), cudaMemcpyHostToDevice);

    solve(d_logits, d_topk_weights, d_topk_indices, M, E, K);

    cudaMemcpy(h_topk_weights, d_topk_weights, M * K * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_topk_indices, d_topk_indices, M * K * sizeof(int), cudaMemcpyDeviceToHost);

    printf("topk_weights:\n");
    for (int r = 0; r < M; r++) {
        printf("[");
        for (int k = 0; k < K; k++) {
            printf("%.6f%s", h_topk_weights[r * K + k], (k < K - 1) ? ", " : "");
        }
        printf("]\n");
    }

    printf("topk_indices:\n");
    for (int r = 0; r < M; r++) {
        printf("[");
        for (int k = 0; k < K; k++) {
            printf("%d%s", h_topk_indices[r * K + k], (k < K - 1) ? ", " : "");
        }
        printf("]\n");
    }

    cudaFree(d_logits);
    cudaFree(d_topk_weights);
    cudaFree(d_topk_indices);

    return 0;
}