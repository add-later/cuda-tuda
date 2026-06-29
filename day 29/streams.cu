#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

__global__ void transform(float* input, float* output, int N){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        float x = input[idx];
        output[idx] = sinf(x) * cosf(x) + sqrtf(fabsf(x));
    }
}

int main() {
    const size_t N = 1'000'000'000ULL;
    const int CHUNK = 1 << 22;
    const int NUM_STREAMS = 4;
    const int THREADS = 256;

    const size_t chunkBytes = (size_t)CHUNK * sizeof(float);

    float* h_input [NUM_STREAMS];
    float* h_output [NUM_STREAMS];
    for (int i = 0; i< NUM_STREAMS; i++){
        cudaMallocHost(&h_input[i], chunkBytes);
        cudaMallocHost(&h_output[i], chunkBytes);
    }

    float* d_input [NUM_STREAMS];
    float* d_output[NUM_STREAMS];
    for (int i = 0; i< NUM_STREAMS; i++){
        cudaMalloc(&d_input[i], chunkBytes);
        cudaMalloc(&d_output[i], chunkBytes);
    }


    cudaStream_t stream[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++){
        cudaStreamCreate(&stream[i]);
    }

    size_t numChunks = (N + CHUNK - 1) / CHUNK;

    for (size_t i = 0; i < numChunks; i++){
        int s = i % NUM_STREAMS;
        size_t offset = i * (size_t)CHUNK;
        int n = (int)((offset + CHUNK <= N) ? CHUNK : (N - offset));

        cudaMemcpyAsync(d_input[s], h_input[s], (size_t)n * sizeof(float), cudaMemcpyHostToDevice, stream[s]);
        
        int blocks = (n + THREADS - 1) / THREADS;
        transform<<<blocks, THREADS, 0, stream[s]>>>(d_input[s], d_output[s], n);

        cudaMemcpyAsync(h_output[s], d_output[s], (size_t)n * sizeof(float), cudaMemcpyDeviceToHost, stream[s]);
    }
    cudaDeviceSynchronize();

    for (int i = 0; i < NUM_STREAMS; i++){
        cudaStreamDestroy(stream[i]);
        cudaFree(d_input[i]);
        cudaFree(d_output[i]);
        cudaFreeHost(h_input[i]);
        cudaFreeHost(h_output[i]);
    }
    return 0;
    }