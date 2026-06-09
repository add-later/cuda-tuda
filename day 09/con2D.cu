#include <iostream>
#include <cuda_runtime.h>

__global__ void solve(const float* input, const float* kernel, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols) {
    int rows = threadIdx.y + blockDim.y * blockIdx.y;
    int cols = threadIdx.x + blockDim.x * blockIdx.x;
    if (rows < input_rows && cols < input_cols) {
        float sum = 0.0;
        for (int i = 0; i < kernel_rows; i++) {
            for (int j = 0; j < kernel_cols; j++) {
                if ((rows + i < input_rows) && (cols + j < input_cols)) {
                    sum += input[(rows + i) * input_cols + (cols + j)] * kernel[kernel_cols * i + j];
                }
            }
        }
        output[input_cols * rows + cols] = sum;
    }
}

int main() {
    using namespace std;

    int input_rows = 3;
    int input_cols = 3;
    int kernel_rows = 2;
    int kernel_cols = 2;

    float h_input[input_rows * input_cols] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    };

    float h_kernel[kernel_rows * kernel_cols] = {
        0.0f, 1.0f,
        1.0f, 0.0f
    };

    float h_output[input_rows * input_cols] = {0.0f};

    size_t input_bytes = input_rows * input_cols * sizeof(float);
    size_t kernel_bytes = kernel_rows * kernel_cols * sizeof(float);
    size_t output_bytes = input_rows * input_cols * sizeof(float);

    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, input_bytes);
    cudaMalloc(&d_kernel, kernel_bytes);
    cudaMalloc(&d_output, output_bytes);

    cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (input_cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (input_rows + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    solve<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_kernel, d_output, input_rows, input_cols, kernel_rows, kernel_cols);

    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost);

    cout << "Output Matrix (" << input_rows << "x" << input_cols << "):\n";
    for (int i = 0; i < input_rows; i++) {
        for (int j = 0; j < input_cols; j++) {
            cout << h_output[i * input_cols + j] << "\t";
        }
        cout << "\n";
    }
    
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    return 0;
}