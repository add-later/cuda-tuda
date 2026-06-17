#include <iostream>
#include <cuda_runtime.h>
#define TILE_WIDTH 16
__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
   __shared__ float tile[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // READING
    if (row < rows && col < cols) {
        tile[ty][tx]= input[row * cols + col];
    }
    __syncthreads();
    int new_row = blockIdx.y * TILE_WIDTH + tx;
    int new_col = blockIdx.x * TILE_WIDTH + ty;

    // CALCULATION
    if (new_row < rows && new_col < cols){
        output[new_col * rows + new_row] = tile[tx][ty];
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