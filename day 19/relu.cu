#include <iostream>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int N) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < N){
        if (input[i]<0){
            output[i] = 0;
        }
        else {
            output[i] = input[i];
        }
    }
}

int main(){
    using namespace std;
    float h_a[5] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    float h_b[5] = {0.0f};
    size_t size = sizeof(float) * 5;

    float *d_a, *d_b;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (5 + threadsPerBlock - 1) / threadsPerBlock;
    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, 5);

    cudaDeviceSynchronize();
    cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 5; i++){
        cout << h_b[i] << " ";
    }

    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}