#include <iostream>
#include <cuda_runtime.h>

#include <iostream>
#include <cuda_runtime.h>

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    if (row < rows && col < cols){
        output[col * rows + row] = input[row * cols + col];
    }
}

int main() {
    using namespace std;
    const int rows = 2;
    const int cols = 3;
    const int size = rows * cols;
    const size_t bytes = size * sizeof(float);

    float h_input[size] = {1.0f, 2.0f, 3.0f, 
                           4.0f, 5.0f, 6.0f};
    float h_output[size] = {0};

    cout << "Input Matrix (2x3):\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            cout << h_input[i * cols + j] << " ";
        }
        cout << "\n";
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16); 
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, 
                  (rows + blockSize.y - 1) / blockSize.y);
    
    matrix_transpose_kernel<<<gridSize, blockSize>>>(d_input, d_output, rows, cols);

    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    cout << "Transposed Matrix (3x2):\n";
    for (int i = 0; i < cols; ++i) {
        for (int j = 0; j < rows; ++j) {
            cout << h_output[i * rows + j] << " ";
        }
        cout << "\n";
    }
    return 0;
}