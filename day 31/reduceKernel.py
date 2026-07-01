import torch
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <cuda_runtime.h>
#define BLOCK_SIZE 256

__global__ void reduceKernel(const float* input, float* output, int N) {
    __shared__ float shared[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    shared[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&output[0], shared[0]);
    }
}

torch::Tensor reduce_sum(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    input = input.contiguous();

    int N = input.numel();
    auto output = torch::zeros({1}, input.options());

    int threads_per_block = BLOCK_SIZE;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    reduceKernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), N
    );

    return output;
}
"""

cpp_source = "torch::Tensor reduce_sum(torch::Tensor input);"

module = load_inline(
    name="reduce_sum_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["reduce_sum"],
)

# --- usage ---
x = torch.randn(1000, device="cuda", dtype=torch.float32)
result = module.reduce_sum(x)
print(result, x.sum())