#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void gemm_kernel(const half* A, const half* B, half* C, int M, int N, int K, float alpha,
                             float beta){
    int x = blockIdx.x * blockDim.x + threadIdx.x; // row in [0, M)
    int y = blockIdx.y * blockDim.y + threadIdx.y; // col in [0, N)

    if (x < M && y < N){
        float acc = 0.0f; // FP32 accumulator

        for (int i = 0; i < K; i++){
            float a = __half2float(A[x * K + i]);
            float b = __half2float(B[i * N + y]);
            acc += a * b;
        }

        float c_old = __half2float(C[x * N + y]);
        float result = alpha * acc + beta * c_old;

        C[x * N + y] = __float2half(result);
    }
}

extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha,
                       float beta) {
    dim3 blockDim(32, 32);
    dim3 gridDim((M + 31) / 32, (N + 31) / 32);
    gemm_kernel<<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
}

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                 \
        }                                                                        \
    } while (0)

int main() {
    const int M = 2, K = 3, N = 2;
    const float alpha = 1.0f, beta = 0.0f;

    // Host data (row-major)
    std::vector<float> h_A_f = {1.0f, 2.0f, 3.0f,
                                 4.0f, 5.0f, 6.0f};
    std::vector<float> h_B_f = {1.0f, 2.0f,
                                 3.0f, 4.0f,
                                 5.0f, 6.0f};
    std::vector<float> h_C_init_f = {1.0f, 1.0f,
                                      1.0f, 1.0f};

    std::vector<half> h_A(M * K), h_B(K * N), h_C(M * N);
    for (int i = 0; i < M * K; i++) h_A[i] = __float2half(h_A_f[i]);
    for (int i = 0; i < K * N; i++) h_B[i] = __float2half(h_B_f[i]);
    for (int i = 0; i < M * N; i++) h_C[i] = __float2half(h_C_init_f[i]);

    half *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), M * N * sizeof(half), cudaMemcpyHostToDevice));

    solve(d_A, d_B, d_C, M, N, K, alpha, beta);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<half> h_C_result(M * N);
    CUDA_CHECK(cudaMemcpy(h_C_result.data(), d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost));

    printf("Result matrix C (%d x %d):\n", M, N);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float val = __half2float(h_C_result[i * N + j]);
            printf("%8.4f ", val);
        }
        printf("\n");
    }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
