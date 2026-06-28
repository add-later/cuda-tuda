#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <iostream>

__global__ void ell_spmv(int num_rows, int K,
                         const int*   ell_cols,
                         const float* ell_vals,
                         const float* x,
                         float*       y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        int   col = ell_cols[k * num_rows + row];
        float val = ell_vals[k * num_rows + row];
        sum += val * x[col];
    }
    y[row] = sum;
}

__global__ void coo_spmv(int num_entries,
                         const int*   coo_rows,
                         const int*   coo_cols,
                         const float* coo_vals,
                         const float* x,
                         float*       y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_entries) return;
    int   row = coo_rows[i];
    int   col = coo_cols[i];
    float val = coo_vals[i];
    atomicAdd(&y[row], val * x[col]);
}

int main()
{
    using namespace std;
    const int num_rows = 4;
    const int num_cols = 4;
    const int K = 2;

    vector<int>   h_ell_cols = {0, 1, 0, 2,
                                     1, 2, 3, 3};
    vector<float> h_ell_vals = {1.f, 3.f, 5.f, 7.f,
                                     2.f, 4.f, 6.f, 8.f};

    const int num_entries = 2;
    vector<int>   h_coo_rows = {0, 3};
    vector<int>   h_coo_cols = {3, 0};
    vector<float> h_coo_vals = {9.f, 10.f};

    vector<float> h_x = {1.f, 1.f, 1.f, 1.f};
    vector<float> h_y(num_rows, 0.f);

    int   *d_ell_cols, *d_coo_rows, *d_coo_cols;
    float *d_ell_vals, *d_coo_vals, *d_x, *d_y;

    cudaMalloc(&d_ell_cols, K * num_rows * sizeof(int));
    cudaMalloc(&d_ell_vals, K * num_rows * sizeof(float));
    cudaMalloc(&d_coo_rows, num_entries  * sizeof(int));
    cudaMalloc(&d_coo_cols, num_entries  * sizeof(int));
    cudaMalloc(&d_coo_vals, num_entries  * sizeof(float));
    cudaMalloc(&d_x, num_cols * sizeof(float));
    cudaMalloc(&d_y, num_rows * sizeof(float));

    cudaMemcpy(d_ell_cols, h_ell_cols.data(), K * num_rows * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_ell_vals, h_ell_vals.data(), K * num_rows * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_coo_rows, h_coo_rows.data(), num_entries  * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_coo_cols, h_coo_cols.data(), num_entries  * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_coo_vals, h_coo_vals.data(), num_entries  * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), num_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, num_rows * sizeof(float));

    const int block = 256;

    ell_spmv<<<(num_rows   + block - 1) / block, block>>>(
        num_rows, K, d_ell_cols, d_ell_vals, d_x, d_y);

    coo_spmv<<<(num_entries + block - 1) / block, block>>>(
        num_entries, d_coo_rows, d_coo_cols, d_coo_vals, d_x, d_y);

    cudaMemcpy(h_y.data(), d_y, num_rows * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_rows; ++i)
        cout<<h_y[i]<<" ";

    cudaFree(d_ell_cols);
    cudaFree(d_ell_vals);
    cudaFree(d_coo_rows);
    cudaFree(d_coo_cols);
    cudaFree(d_coo_vals);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
