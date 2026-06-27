#include <iostream>
#include <cuda_runtime.h>
#include <iostream>
__global__ void clip_kernel(const float* input, float* output, float lo, float hi, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N){
        if (input[idx] < lo){
            output[idx] = lo;
        } else if (input[idx] > hi){
            output[idx] = hi;
        } else {
            output[idx] = input[idx];
        }
    }
}

int main(){
    using namespace std;
    int N = 4;
    float h_input[N] = {1.5f, -2.0f, 3.0f, 4.5f};
    float lo = 0.0f;
    float hi = 3.5f;

    float size = N * sizeof(float);

    float *h_output = (float*)malloc(N * sizeof(float));   // or just: malloc(size)

    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    clip_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, lo, hi, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++){
        cout << h_output[i] << " ";
    }

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_output);

    return 0;
}