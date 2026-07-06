#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>

using namespace nvcuda;

#define M 16
#define N 16
#define K 16

__global__ void wmma_matmul_fp16(half *a, half *b, float *c)
{
    wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N, K, float>              c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    wmma::load_matrix_sync(a_frag, a, K);
    wmma::load_matrix_sync(b_frag, b, N);

    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    wmma::store_matrix_sync(c, c_frag, N, wmma::mem_row_major);
}


int main()
{
    size_t a_bytes = M * K * sizeof(half);
    size_t b_bytes = K * N * sizeof(half);
    size_t c_bytes = M * N * sizeof(float);

    half  *h_a = (half*) malloc(a_bytes);
    half  *h_b = (half*) malloc(b_bytes);
    float *h_c = (float*)malloc(c_bytes);

    for (int i = 0; i < M * K; i++) h_a[i] = __float2half(1.0f);
    for (int i = 0; i < K * N; i++) h_b[i] = __float2half(1.0f);

    half  *d_a, *d_b;
    float *d_c;
    cudaMalloc(&d_a, a_bytes);
    cudaMalloc(&d_b, b_bytes);
    cudaMalloc(&d_c, c_bytes);

    cudaMemcpy(d_a, h_a, a_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, b_bytes, cudaMemcpyHostToDevice);

    wmma_matmul_fp16<<<1, 32>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, c_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    free(h_a);
    free(h_b);
    free(h_c);
    return 0;
}
