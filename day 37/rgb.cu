#include <cuda_runtime.h>
#include <iostream> 
using namespace std;
__global__ void rgb_to_grayscale_kernel(const float* input, float* output, int width, int height) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if ((idx < width * height)){
        int b = idx * 3;
        output[idx] = 0.299 * input[b] + 0.587 * input[b + 1] + 0.114 * input[b + 2];
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int width, int height) {
    int total_pixels = width * height;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;

    rgb_to_grayscale_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, width, height);
    cudaDeviceSynchronize();
}

int main() {
    const int width = 2;
    const int height = 2;

    float A[width * height * 3] = {255.0, 0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 255.0, 128.0, 128.0, 128.0};
    float output[width * height];

    float *d_A, *d_output;

    cudaMalloc((void**)&d_A, width * height * 3 * sizeof(float));
    cudaMalloc((void**)&d_output, width * height * sizeof(float));

    cudaMemcpy(d_A, A, width * height * 3 * sizeof(float), cudaMemcpyHostToDevice);

    solve(d_A, d_output, width, height);

    cudaMemcpy(output, d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    cout << "Output: [";
    for (int i = 0; i < width * height; i++) {
        cout << output[i];
    }
    cout << "]" << endl;

    cudaFree(d_A);
    cudaFree(d_output);

    return 0;
}

