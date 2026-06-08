#include <iostream>
#include <cuda_runtime.h>
__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < input_size - kernel_size + 1) {
        float sum = 0.0;
        for (int j = 0; j < kernel_size; j++) {
            sum += input[idx + j] * kernel[j];
        }
        output[idx] = sum;
    }
}

int main() {
    using namespace std;
    int input_size = 5;
    int kernel_size = 3;
    int output_size = input_size - kernel_size + 1;

    float h_input[input_size] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float h_kernel[kernel_size] = {1.0f, 0.0f, -1.0f};
    float h_output[output_size] = {0.0f};

    float *d_input, *d_kernel, *d_output;
    cudaMalloc((void**)&d_input, input_size * sizeof(float));
    cudaMalloc((void**)&d_kernel, kernel_size * sizeof(float));
    cudaMalloc((void**)&d_output, output_size * sizeof(float));

    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_kernel, d_output, input_size, kernel_size);

    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    cout << "Output: [";
    for (int i = 0; i < output_size; ++i) {
        cout << h_output[i];
        if (i < output_size - 1){
            cout << ", ";
    }
    }
    cout << "]" << std::endl;

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    return 0;
}