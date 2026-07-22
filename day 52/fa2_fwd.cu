#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#define CUDA_CHECK(call) \
do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess){ \
        fprintf(stderr, "CUDA error %s at %s: %d\n", \
        cudaGetErrorString(err__), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

template <int HEAD_DIM, int BLOCK_ROWS_Q, int BLOCK_ROWS_KV>

__global__ void flash_attn_fwd(
    const float* __restrict__ Q, 
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    float* __restrict__ L,
    int N, float scale 
) {
    const int q_tile_idx = blockIdx.x;
    const int batch_head_idx = blockIdx.y;
    const int q_row_in_tile = threadIdx.x;

    const int q_row_global = q_tile_idx * BLOCK_ROWS_Q + q_row_in_tile;

    const float* Q_head = Q + (size_t)batch_head_idx * N * HEAD_DIM;
    const float* K_head = K + (size_t)batch_head_idx * N * HEAD_DIM;
    const float* V_head = V + (size_t)batch_head_idx * N * HEAD_DIM;
    float* O_head = O + (size_t)batch_head_idx * N * HEAD_DIM;
    float* L_head = L + (size_t)batch_head_idx * N;

    extern __shared__ float smem[];
    float* sQ = smem;
    float* sK = sQ + BLOCK_ROWS_Q * HEAD_DIM;
    float* sV = sK + BLOCK_ROWS_KV * HEAD_DIM;

    if (q_row_global < N) {
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d)
            sQ[q_row_in_tile * HEAD_DIM + d] = Q_head[(size_t)q_row_global * HEAD_DIM + d];
    } else {
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d)
            sQ[q_row_in_tile * HEAD_DIM + d] = 0.0f;
    }

    float m_i = -INFINITY;
    float l_i = 0.0f;
    float acc[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++){
        acc[d] = 0.0f;
    }

    const int num_kv_tiles = (N + BLOCK_ROWS_KV - 1) / BLOCK_ROWS_KV; //T_c

    for (int kv_tile_idx = 0; kv_tile_idx < num_kv_tiles; kv_tile_idx++){
        const int kv_row_start = kv_tile_idx * BLOCK_ROWS_KV;
        const int valid_kv_rows = min(BLOCK_ROWS_KV, N - kv_row_start);

        __syncthreads();

        for(int flat = q_row_in_tile; flat < BLOCK_ROWS_KV * HEAD_DIM; flat += BLOCK_ROWS_Q){
            int kv_row = flat / HEAD_DIM; //rowin KV tile
            int d = flat % HEAD_DIM; //col idx
            int global_row = kv_row_start + kv_row;

            if (global_row < N){
                sK[flat] = K_head[(size_t)global_row * HEAD_DIM + d];
            } else {
                sK[flat] = 0.0f;
            }

            if (global_row < N){
                sV[flat] = V_head[(size_t)global_row * HEAD_DIM + d];
            } else {
                sV[flat] = 0.0f;
            }
        }
        __syncthreads();

        float S_row[BLOCK_ROWS_KV];
        float row_max = -INFINITY;
        #pragma unroll
        for (int kv_row = 0; kv_row < BLOCK_ROWS_KV; kv_row++){
            if (kv_row < valid_kv_rows) {
                float dot = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; d++){
                    dot += sQ[q_row_in_tile * HEAD_DIM + d] * sK[kv_row * HEAD_DIM + d];
                }
                dot *= scale;
                S_row[kv_row] = dot;
                row_max = fmaxf(row_max, dot);
            } else {
                S_row[kv_row] = -INFINITY;
            }
        }

        const float m_new = fmaxf(m_i, row_max);
        const float correction = expf(m_i - m_new);

        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++){
            acc[d] *= correction;
        }
        
        float rowsum_p = 0.0f;
        #pragma unroll
        for (int kv_row = 0; kv_row < BLOCK_ROWS_KV; kv_row++){
            float p = expf(S_row[kv_row] - m_new);
            rowsum_p += p;

            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++){
                acc[d] += p * sV[kv_row * HEAD_DIM + d];
            }
        }

        l_i = l_i * correction + rowsum_p;
        m_i = m_new;
    }

    if (q_row_global < N){
        const float inv_l_i = 1.0f / l_i;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++){
            O_head[(size_t)q_row_global * HEAD_DIM + d] = acc[d] * inv_l_i;
        }
        L_head[q_row_global] = m_i + logf(l_i);
    }
}

template <int HEAD_DIM, int BLOCK_ROWS_Q, int BLOCK_ROWS_KV>
void launch_flash_attention2_forward(
    const float* d_Q, const float* d_K, const float* d_V,
    float* d_O, float* d_L, int batch_heads, int N, float scale
){
    const int num_q_tiles = (N + BLOCK_ROWS_Q - 1) / BLOCK_ROWS_Q;
    dim3 grid(num_q_tiles, batch_heads);
    dim3 block(BLOCK_ROWS_Q);

    size_t smem_bytes = (size_t)(BLOCK_ROWS_Q * HEAD_DIM + 2 * BLOCK_ROWS_KV * HEAD_DIM) * sizeof(float);

    flash_attn_fwd<HEAD_DIM, BLOCK_ROWS_Q, BLOCK_ROWS_KV><<<grid, block, smem_bytes>>>(d_Q, d_K, d_V, d_O, d_L, N, scale);
    CUDA_CHECK(cudaGetLastError());
}

void cpu_attention_reference(
    const std::vector<float>& Q, const std::vector<float>& K, const std::vector<float>& V,
    std::vector<float>& O, int batch_heads, int N, int head_dim, float scale)
{
    O.assign((size_t)batch_heads * N * head_dim, 0.0f);
    std::vector<float> scores(N);

    for (int bh = 0; bh < batch_heads; ++bh) {
        const float* Q_head = &Q[(size_t)bh * N * head_dim];
        const float* K_head = &K[(size_t)bh * N * head_dim];
        const float* V_head = &V[(size_t)bh * N * head_dim];
        float*       O_head = &O[(size_t)bh * N * head_dim];

        for (int q_row = 0; q_row < N; ++q_row) {
            float row_max = -INFINITY;
            for (int kv_row = 0; kv_row < N; ++kv_row) {
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d)
                    dot += Q_head[q_row * head_dim + d] * K_head[kv_row * head_dim + d];
                dot *= scale;
                scores[kv_row] = dot;
                row_max = std::max(row_max, dot);
            }
            float denom = 0.0f;
            for (int kv_row = 0; kv_row < N; ++kv_row) {
                scores[kv_row] = expf(scores[kv_row] - row_max);
                denom += scores[kv_row];
            }
            for (int d = 0; d < head_dim; ++d) {
                float acc = 0.0f;
                for (int kv_row = 0; kv_row < N; ++kv_row)
                    acc += scores[kv_row] * V_head[kv_row * head_dim + d];
                O_head[q_row * head_dim + d] = acc / denom;
            }
        }
    }
}

float* upload_to_device(const std::vector<float>& host_data) {
    float* device_ptr;
    size_t bytes = host_data.size() * sizeof(float);
    CUDA_CHECK(cudaMalloc(&device_ptr, bytes));
    CUDA_CHECK(cudaMemcpy(device_ptr, host_data.data(), bytes, cudaMemcpyHostToDevice));
    return device_ptr;
}

int main() {
    constexpr int HEAD_DIM = 64;
    constexpr int BLOCK_ROWS_Q = 64;
    constexpr int BLOCK_ROWS_KV= 64;

    const int batch = 2;
    const int heads = 4;
    const int N = 256;
    const int batch_heads = batch * heads;
    const float scale = 1.0f / sqrtf((float)HEAD_DIM);
    const size_t qkv_elems = (size_t)batch_heads * N * HEAD_DIM;

    std::vector<float> h_Q(qkv_elems), h_K(qkv_elems), h_V(qkv_elems);
    std::mt19937 rng(0);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : h_Q) v = dist(rng);
    for (auto& v : h_K) v = dist(rng);
    for (auto& v : h_V) v = dist(rng);

    float* d_Q = upload_to_device(h_Q);
    float* d_K = upload_to_device(h_K);
    float* d_V = upload_to_device(h_V);
    float* d_O;
    float* d_L;
    CUDA_CHECK(cudaMalloc(&d_O, qkv_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_L, (size_t)batch_heads * N * sizeof(float)));

    launch_flash_attention2_forward<HEAD_DIM, BLOCK_ROWS_Q, BLOCK_ROWS_KV>(
        d_Q, d_K, d_V, d_O, d_L, batch_heads, N, scale);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_O(qkv_elems);
    CUDA_CHECK(cudaMemcpy(h_O.data(), d_O, qkv_elems * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> ref_O;
    cpu_attention_reference(h_Q, h_K, h_V, ref_O, batch_heads, N, HEAD_DIM, scale);

    double max_abs_err = 0.0;
    for (size_t i = 0; i < h_O.size(); ++i)
        max_abs_err = std::max(max_abs_err, (double)fabs(h_O[i] - ref_O[i]));

    printf("Max abs error vs CPU reference: %.6e\n", max_abs_err);
    printf(max_abs_err < 1e-3 ? "PASS\n" : "FAIL\n");

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O); cudaFree(d_L);
    return 0;
}
