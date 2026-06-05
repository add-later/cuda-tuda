#include <iostream>
#include <cuda_runtime.h>

#include <iostream>
#include <cuda_runtime.h>

__global__ void layerNorm(const float* input, int N, float* gamma, float* beta, float* output ){
    __shared__ float arr[1024];
    __shared__ float arr_sq[1024];
    float sum = 0.0;
    float sum_sq = 0.0;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    for (int i = idx; i < N; i += blockDim.x) {
        sum += input[i];
        sum_sq += input[i] * input[i];
        }
    for (int offset = 32/2; offset > 0; offset /= 2){
        sum += __shfl_down_sync(0xffffffff, sum, offset); 
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }
    __syncthreads();
    if (threadIdx.x % 32 == 0){
          arr[threadIdx.x/32] = sum;
          arr_sq[threadIdx.x/32] = sum_sq;
    }
    int stride = blockDim.x/32;
    __syncthreads();
    while (stride != 0){
        if (threadIdx.x < stride){
            arr[threadIdx.x] += arr[threadIdx.x + stride];
            arr_sq[threadIdx.x] += arr_sq[threadIdx.x + stride];
        }
        __syncthreads();
        stride /= 2;
        
    }
    
    float mean = arr[0] / N;
    float variance = arr_sq[0] / N - mean * mean;

    for (int i = idx; i < N; i += blockDim.x) {
        output[i] = (input[i] - mean) / sqrt(variance + 1e-5) * gamma[i] + beta[i];
    }
    
}

int main() {
    using namespace std;

    const int N = 1024;
    const int BLOCK_DIM = 256;
    const int GRID_DIM = 1;

    float* h_input  = new float[N];
    float* h_gamma  = new float[N];
    float* h_beta   = new float[N];
    float* h_output = new float[N];

    for (int i = 0; i < N; i++) {
        h_input[i] = static_cast<float>(i) / N;
        h_gamma[i] = 1.0f;
        h_beta[i]  = 0.0f;
    }

    float *d_input, *d_gamma, *d_beta, *d_output;
    cudaMalloc(&d_input,  N * sizeof(float));
    cudaMalloc(&d_gamma,  N * sizeof(float));
    cudaMalloc(&d_beta,   N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    cudaMemcpy(d_input,  h_input,  N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma,  h_gamma,  N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta,   h_beta,   N * sizeof(float), cudaMemcpyHostToDevice);

    layerNorm<<<GRID_DIM, BLOCK_DIM>>>(d_input, N, d_gamma, d_beta, d_output);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    float mean = 0.0f, var = 0.0f;
    for (int i = 0; i < N; i++) mean += h_input[i];
    mean /= N;
    for (int i = 0; i < N; i++) var += (h_input[i] - mean) * (h_input[i] - mean);
    var /= N;

    float max_err = 0.0f;
    for (int i = 0; i < N; i++) {
        float ref = (h_input[i] - mean) / sqrtf(var + 1e-5f) * h_gamma[i] + h_beta[i];
        float err = fabsf(h_output[i] - ref);
        if (err > max_err) max_err = err;
    }
    printf("Max absolute error vs CPU reference: %e\n", max_err);

    printf("First 8 outputs:\n");
    
    cudaFree(d_input);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_gamma;
    delete[] h_beta;
    delete[] h_output;

    return 0;
}