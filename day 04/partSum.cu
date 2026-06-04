#include <iostream>
#include <cuda_runtime.h>

__global__ void partial_sum(const float* input, float* output, int N){
    __shared__ float arr[1024];
    float sum = 0.0;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    for (int i = idx; i < N; i += 1024){
        sum += input[i];
    }
    arr[threadIdx.x] = sum;
    __syncthreads();
    int stride = 1024/2;
    while (stride != 0){
        if (threadIdx.x < stride){
            arr[threadIdx.x] += arr[threadIdx.x + stride];
        }
        __syncthreads();
        stride /= 2;
    }
    if (threadIdx.x == 0){
        output[0] = arr[0];
    }
}

int main() {
    using namespace std;

    const int N = 8;
    float h_input[N] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float h_output = 0.0f;

    cout << "[";
    for (int i = 0; i < N; ++i) {
        cout << h_input[i] << (i < N - 1 ? ", " : "");
    }
    cout << "]\n";

    float* d_input = nullptr;
    float* d_output = nullptr;

    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, &h_output, sizeof(float), cudaMemcpyHostToDevice);

    partial_sum<<<1, 1024>>>(d_input, d_output, N);

    cudaDeviceSynchronize();

    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    
    cout << "Output: " << h_output << endl;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}