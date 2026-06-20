#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <iostream>
#include <device_launch_parameters.h>

void verify_solution(float *a, float *b, float *c, int n){
    float temp;
    float epsilon = 0.001f;
    for (int row = 0; row < n; row++){
        for (int col = 0; col < n; col++){
            temp = 0.0f;
            for (int shared_dim = 0; shared_dim < n; shared_dim++){
                temp += a[shared_dim * n + row] * b[col * n + shared_dim];
            }
            assert(fabs(c[col * n + row] - temp) < epsilon);
        }
    }
}

int main() {
  using namespace std;
  int n = 1 << 10;
  size_t bytes = n * n * sizeof(float);

  float *h_a, *h_b, *h_c;
  float *d_a, *d_b, *d_c;

  h_a = (float*)malloc(bytes);
  h_b = (float*)malloc(bytes);
  h_c = (float*)malloc(bytes);

  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  curandGenerator_t prng;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());

  curandGenerateUniform(prng, d_a, n * n);
  curandGenerateUniform(prng, d_b, n * n);

  cublasHandle_t handle;
  cublasCreate(&handle);

  // CUBLAS_OP_N is for non-transposed matrices
  // n, n, n are dimensions of matrices 
  // alpha, beta are scaling factors (implementation is different) C = alpha * op(A) * op(B) + beta * C

  float alpha = 1.0f;
  float beta = 0.0f;

  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_a, n, d_b, n, &beta, d_c, n);

  cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

  verify_solution(h_a, h_b, h_c, n);

  cublasDestroy(handle);
  curandDestroyGenerator(prng);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  free(h_a); 
  free(h_b); 
  free(h_c);

  cout<<"IT WORKS!!!";
  return 0;
}