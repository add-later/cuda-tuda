#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <vector>

#define BLOCK_Q 32
#define BLOCK_K 32
#define MAX_D 128

__global__ void flash_atten_fwd_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    float* __restrict__ L,
    const int N,
    const int d,
    const float scale,
    const bool causal
) {
    extern __shared__ float shared_mem[];
    float* Kt = shared_mem;
    float* Vt = shared_mem + BLOCK_K * d;

    const int bh = blockIdx.y;
    const float* Q_block = Q + (size_t)bh * N * d;
    const float* K_block = K + (size_t)bh * N * d;
    const float* V_block = V + (size_t)bh * N * d;
    float* O_block = O + (size_t)bh * N * d;
    float* L_block = L + (size_t)bh * N;

    const int q_block_start = blockIdx.x * BLOCK_Q;
    const int q_block_end = min(q_block_start + BLOCK_Q - 1, N - 1);
    const int tid = threadIdx.x;
    const int q_row = q_block_start + tid;

    float row_max = -FLT_MAX;
    float row_sum = 0.0f;

    float acc[MAX_D];
    float q_req[MAX_D];

    for (int i = 0; i < d; i++) {
        acc[i] = 0.0f;
    }

    if (q_row < N) {
        for (int i = 0; i < d; i++) {
            q_req[i] = Q_block[q_row * d + i];
        }
    }

    int k_loop_end;
    if (causal) {
        k_loop_end = q_block_end + 1;
    } else {
        k_loop_end = N;
    }

    for (int k_block_start = 0; k_block_start < k_loop_end; k_block_start += BLOCK_K) {
        int k_rows = min(BLOCK_K, k_loop_end - k_block_start);

        for (int r = tid; r < k_rows; r += BLOCK_Q) {
            for (int i = 0; i < d; i++) {
                Kt[r * d + i] = K_block[(k_block_start + r) * d + i];
                Vt[r * d + i] = V_block[(k_block_start + r) * d + i];
            }
        }

        __syncthreads();

        if (q_row < N) {
            const bool causal_mask = causal && (k_block_start + k_rows - 1 > q_block_start);
            float scores[BLOCK_K];
            float block_max = -FLT_MAX;

            for (int i = 0; i < k_rows; i++) {
                int k_row = k_block_start + i;
                if (causal_mask && k_row > q_row) {
                    scores[i] = -FLT_MAX;
                    continue;
                }
                float s = 0.0f;
                for (int j = 0; j < d; j++) {
                    // Read from the shared-memory tile (Kt), not global K_block.
                    s += q_req[j] * Kt[i * d + j];
                }
                s *= scale;
                scores[i] = s;
                block_max = fmaxf(block_max, s);
            }

            if (block_max > -FLT_MAX) {
                float new_max = fmaxf(row_max, block_max);
                float correction = expf(row_max - new_max);

                row_sum *= correction;
                for (int i = 0; i < d; i++) {
                    acc[i] *= correction;
                }

                for (int i = 0; i < k_rows; i++) {
                    if (scores[i] == -FLT_MAX) continue;
                    float p = expf(scores[i] - new_max);
                    row_sum += p;
                    for (int j = 0; j < d; j++) {
                        acc[j] += p * Vt[i * d + j];
                    }
                }
                row_max = new_max;
            }
        }
        __syncthreads();
    }

    if (q_row < N) {
        for (int i = 0; i < d; i++) {
            O_block[q_row * d + i] = acc[i] / row_sum;
        }
        L_block[q_row] = row_max + logf(row_sum);
    }
}

std::vector<torch::Tensor> flash_attn_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    bool causal
) {
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat32, "only float32 is supported");
    TORCH_CHECK(Q.dim() == 3, "Q must be (batch_heads, N, d)"); // flatten batch*heads before calling

    const int BH = Q.size(0);
    const int N = Q.size(1);
    const int d = Q.size(2);
    TORCH_CHECK(d <= MAX_D, "head_dim exceeds MAX_D (", MAX_D, ")");

    const float scale = 1.0f / sqrtf((float)d);

    auto O = torch::empty({BH, N, d}, Q.options());
    auto L = torch::empty({BH, N}, Q.options());

    const int num_q_blocks = (N + BLOCK_Q - 1) / BLOCK_Q;
    const dim3 grid(num_q_blocks, BH);
    const int threads_per_block = BLOCK_Q;
    const size_t shared_bytes = 2 * (size_t)BLOCK_K * d * sizeof(float);

    flash_atten_fwd_kernel<<<grid, threads_per_block, shared_bytes>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        O.data_ptr<float>(), L.data_ptr<float>(), N, d, scale, causal
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return {O, L};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_forward", &flash_attn_forward, "Flash attention forward (CUDA)");
}