#include <cuda_runtime.h>
#include <iostream>

__device__ unsigned int fnv1a_hash(unsigned int input) {
    const unsigned int FNV_PRIME = 16777619;
    const unsigned int OFFSET_BASIS = 2166136261;

    unsigned int hash = OFFSET_BASIS;
    for (int byte_pos = 0; byte_pos < 4; byte_pos++) {
        unsigned char byte = (input >> (byte_pos * 8)) & 0xFFu;
        hash = (hash ^ byte) * FNV_PRIME;
    }
    return hash;
}

__global__ void fnv1a_hash_kernel(const int* input, unsigned int* output, int N, int R) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) {
        int temp = input[idx];
        for (int i = 0; i < R; i++)
            temp = fnv1a_hash(temp);
        output[idx] = temp;
    }
}

void solve(const int* input, unsigned int* output, int N, int R) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    fnv1a_hash_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, R);
    cudaDeviceSynchronize();
}

int main() {
    int N = 3, R = 2;
    int   h_input[]  = {123, 456, 789};
    unsigned int h_output[3] = {};

    int          *d_input;
    unsigned int *d_output;

    cudaMalloc(&d_input,  N * sizeof(int));
    cudaMalloc(&d_output, N * sizeof(unsigned int));

    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    solve(d_input, d_output, N, R);

    cudaMemcpy(h_output, d_output, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    printf("Output: ");
    for (int i = 0; i < N; i++)
        printf("%u%s", h_output[i], i < N-1 ? ", " : "\n");

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}