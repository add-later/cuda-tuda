#include <iostream>
#include <cuda_runtime.h>

__global__ void dot_product_kernel(const float* a, const float* b, float* c, int n) {
    int i = threadIdx.x;
    if (i < n) {
        c[i] = a[i] * b[i];
    }
}

int main() {
    using namespace std;

    const int N = 3;
    size_t size = N * sizeof(float);

    float h_a[N] = {1.0f, 2.0f, 3.0f};
    float h_b[N] = {4.0f, 5.0f, 6.0f};
    float h_c[N] = {0.0f, 0.0f, 0.0f}; // Holds intermediate products

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    dot_product_kernel<<<1, N>>>(d_a, d_b, d_c, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    float final_dot_product = 0.0f;
    for (int i = 0; i < N; i++) {
        final_dot_product += h_c[i];
    }

    cout << "a = [1, 2, 3], b = [4, 5, 6]" << endl;
    cout << "-> " << h_a[0] << "*" << h_b[0] << " + "
         << h_a[1] << "*" << h_b[1] << " + "
         << h_a[2] << "*" << h_b[2] << " = " 
         << final_dot_product << endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}