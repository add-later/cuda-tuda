#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void bmm_fp16(const half* __restrict__ A, 
                         const half* __restrict__ B,
                         half* __restrict__ C,
                         int Bt, 
                         int M, 
                         int K, 
                         int N){
    int batch = blockIdx.z;
    if (batch >= Bt) return;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = (blockIdx.x * blockDim.x + threadIdx.x) / 32;

    int tileRow = row * WMMA_M;
    int tileCol = col * WMMA_N;

    if (row > M || col > N) return;

    const half* A_ptr = A + (size_t)batch * M * K;
    const half* B_ptr = B + (size_t)batch * K * N;
    half* C_ptr = C + (size_t)batch * M * N;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    __shared__ half As[WMMA_M][WMMA_K];
    __shared__ half Bs[WMMA_K][WMMA_N];

    int lane = threadIdx.x % 32;

    for (int k = 0; k < K; k += WMMA_K){
        int k_r = K - k;
        if (k_r >= WMMA_K && tileRow + WMMA_M <= M && tileCol + WMMA_N <= N){
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;

            wmma::load_matrix_sync(a_frag, A_ptr + tileRow * K + k, K);
            wmma::load_matrix_sync(b_frag, B_ptr + k * N + tileCol, N);

            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        } else {
            for (int i = lane; i < WMMA_M * WMMA_K; i+= 32){
                int r = i / WMMA_K;
                int c = i % WMMA_K;
                int gr = tileRow + r;
                int gc = k + c;
                As[r][c] = (gr < M && gc < K) ? A_ptr[gr * K + gc] : __float2half(0.0f);
            }
            for (int i = lane; i < WMMA_K * WMMA_N; i += 32){
                int r = i / WMMA_N;
                int c = i % WMMA_N;
                int gr = k + r;
                int gc = tileCol + c;
                Bs[r][c] = (gr < K && gc < N) ? B_ptr[gr * N + gc] : __float2half(0.0f);
            }
            __syncwarp();
            
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
            wmma::load_matrix_sync(a_frag, &As[0][0], WMMA_K);
            wmma::load_matrix_sync(b_frag, &Bs[0][0], WMMA_N);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            __syncwarp();
        }
        }
    
    if (tileRow + WMMA_N <= M && tileCol + WMMA_M <= N){
        __shared__ float Cs[WMMA_M][WMMA_N];
        wmma::store_matrix_sync(&Cs[0][0], acc_frag, WMMA_N, wmma::mem_row_major);
        for (int i = lane; i < WMMA_M * WMMA_N; i += 32){
            int r = i / WMMA_N;
            int c = i % WMMA_N;
            C_ptr[(tileRow + r) * N + (tileCol + c)] = __float2half(Cs[r][c]);
        }
    } else {
        __shared__ float Cs[WMMA_M][WMMA_N];
        wmma::store_matrix_sync(&Cs[0][0], acc_frag, WMMA_N, wmma::mem_row_major);
        __syncwarp();
        for (int i = lane; i < WMMA_M * WMMA_N; i += 32){
            int r = i / WMMA_N;
            int c = i % WMMA_N;
            int gr = tileRow + r;
            int gc = tileCol + c;
            if (gr < M && gc < N) C_ptr[gr * N + gc] = __float2half(Cs[r][c]);
        }
    }
                         }

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e){ \
  printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
  exit(1);} } while(0)

int main() {
    int Bt = 4, M = 64, K = 64, N = 64;

    size_t szA = (size_t)Bt*M*K, szB = (size_t)Bt*K*N, szC = (size_t)Bt*M*N;
    std::vector<half>  hA(szA), hB(szB), hC(szC);
    std::vector<float> fA(szA), fB(szB);   // keep float copies for the CPU check

    for (size_t i = 0; i < szA; i++) { fA[i] = rand()/(float)RAND_MAX - 0.5f; hA[i] = __float2half(fA[i]); }
    for (size_t i = 0; i < szB; i++) { fB[i] = rand()/(float)RAND_MAX - 0.5f; hB[i] = __float2half(fB[i]); }

    half *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, szA*sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dB, szB*sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dC, szC*sizeof(half)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), szA*sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), szB*sizeof(half), cudaMemcpyHostToDevice));

    // one warp per output tile: blockDim.x = 32, blockDim.y = 1
    dim3 block(32, 1, 1);
    dim3 grid((N + WMMA_N - 1) / WMMA_N,
              (M + WMMA_M - 1) / WMMA_M,
              Bt);

    bmm_fp16<<<grid, block>>>(dA, dB, dC, Bt, M, K, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hC.data(), dC, szC*sizeof(half), cudaMemcpyDeviceToHost));

    double maxErr = 0.0;
    for (int b = 0; b < Bt; b++)
      for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float acc = 0.f;
            for (int k = 0; k < K; k++)
                acc += fA[((size_t)b*M + i)*K + k] * fB[((size_t)b*K + k)*N + j];
            float got = __half2float(hC[((size_t)b*M + i)*N + j]);
            maxErr = fmax(maxErr, fabs(got - acc));
        }
    printf("Max abs error: %g\n", maxErr);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}