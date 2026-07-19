#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

__device__ void heap_down(float* val, int* idx, int k, in size){
    int minimum = k;
    int left= 2 * k + 1;
    int right = 2 * k + 2;

    if (left < size && val[left] < val[minimum]){
        minimum = left;
    }
    if (right < size && val[right] < val[minimum]){
        minimum = right;
    }

    if (minimum != k){
        float thread_value = val[k];
        val[k] = val[minimum];
        val[minimum] = thread_value;

        int thread_index = idx[k];
        idx[k] = idx[minimum];
        idx[minimum] = thread_index;
        heap_down(val, idx, minimum, size);
    }
}

__global__ void top_k(const float* __restrict__ input, 
    int* __restrict__ indices,
    float* __restrict__ output,
    int N, int K){
        int tid = thread_idx.x;

        const float* row_input = input + blockIdx.x * N;
        float* row_output = output + blockIdx.x * K;
        int* row_indices = indices + blockIdx.x * K;

        extern __shared__ unsigned char smem[];
        float* heap_val = reinterpret_cast<float*>(smem);
        int* heap_idx = reinterpret_cast<int*>(smem + K * sizeof(float));

        if (tid < K){
            heap_val[tid] = row_input[tid];
            heap_idx[tid] = tid;
        }
        __syncthreads();

        if (tid == 0){
            for (int i = K/2 - 1; i >= 0; i--){
                heap_down(heap_val, heap_idx, i, K);
            }

            for (int i = K; i < N; i++){
                if (row_input[i] > heap_val[0]){
                    heap_val[0] = row_input[i];
                    heap_idx[0] = i;
                    heap_down(heap_val, heap_idx, 0, K);
                }
            }
        }
        __syncthreads();

        if (tid < K){
            row_output[tid] = heap_val[tid];
            row_indices[tid] = heap_idx[tid];
        }
    }

    void solve(const float* logits, float* topk_weights, int* topk_indices, int M, int E, int k) {
    dim3 block_size(max(k, 32));
    dim3 grid_size(M);

    size_t shmem_bytes = k * sizeof(float) + k * sizeof(int);

    top_k<<<grid_size, block_size, shmem_bytes>>>(logits, topk_indices, topk_weights, E, k);
    cudaDeviceSynchronize();
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