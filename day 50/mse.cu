#include <cuda_runtime.h>
#include <vector>
#include <iostream>

__global__ void mse_kernel(const float* predictions, const float* targets, double* sum, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        double diff = (double)predictions[idx] - (double)targets[idx];
        atomicAdd(sum, diff * diff);
    }
}

extern "C" void solve(const float* predictions, const float* targets, float* mse, int N) {
    double* d_sum;
    cudaMalloc(&d_sum, sizeof(double));
    cudaMemset(d_sum, 0, sizeof(double));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    mse_kernel<<<blocksPerGrid, threadsPerBlock>>>(predictions, targets, d_sum, N);
    cudaDeviceSynchronize();

    double h_sum = 0.0;
    cudaMemcpy(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
    double mse_val = h_sum / (double)N;

    float h_mse = (float)mse_val;
    cudaMemcpy(mse, &h_mse, sizeof(float), cudaMemcpyHostToDevice);

    cudaFree(d_sum);
}

int main() {
    int N = 4;
    std::vector<float> h_predictions = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> h_targets     = {1.5f, 2.5f, 3.5f, 4.5f};

    float *d_predictions, *d_targets, *d_mse;
    cudaMalloc(&d_predictions, N * sizeof(float));
    cudaMalloc(&d_targets, N * sizeof(float));
    cudaMalloc(&d_mse, sizeof(float));

    cudaMemcpy(d_predictions, h_predictions.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, h_targets.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    solve(d_predictions, d_targets, d_mse, N);

    float h_mse;
    cudaMemcpy(&h_mse, d_mse, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << h_mse << std::endl;

    cudaFree(d_predictions);
    cudaFree(d_targets);
    cudaFree(d_mse);

    return 0;
}