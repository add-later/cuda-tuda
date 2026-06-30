#include <cstdio>
#include <cuda_runtime.h>

constexpr int N = 1 << 14;
constexpr int STAGES = 8;
constexpr int ITERS = 200;

__global__ void k_scale(float* input, float k, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        input[i] *= k;
    }
}

__global__ void k_add(float* input, float value, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        input[i] += value;
    }
}

__global__ void k_sqrt(float* input, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N){
        input[i] = sqrtf(fabsf(input[i]));
    }
}

void launch_chain(float* d_x, cudaStream_t stream) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    for (int s = 0; s < STAGES; ++s) {
        k_scale<<<blocks, threads, 0, stream>>>(d_x, 1.0001f, N);
        k_add<<<blocks, threads, 0, stream>>>(d_x, 0.5f, N);
        k_sqrt<<<blocks, threads, 0, stream>>>(d_x, N);
    }
}

int main(){
    float* d_input;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMemset(d_input, 0, N * sizeof(float));
               
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Baseline
    cudaEventRecord(start, stream);
    for (int i = 0; i < ITERS; i++){
        launch_chain(d_input, stream);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float baseline_ms = 0.0;
    cudaEventElapsedTime(&baseline_ms, start, stop);
    printf("Baseline: %f ms\n", baseline_ms);

    // Graph
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    launch_chain(d_input, stream);
    cudaStreamEndCapture(stream, &graph);
    
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

    cudaEventRecord(start, stream);
    for (int i = 0; i < ITERS; i++){
        cudaGraphLaunch(graphExec, stream);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float graph_ms = 0.0;
    cudaEventElapsedTime(&graph_ms, start, stop);
    printf("Graph: %f ms\n", graph_ms);


    size_t numNodes = 0;
    cudaGraphGetNodes(graph, nullptr, &numNodes);
    printf("Number of nodes: %d\n", numNodes);

    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(d_input);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}