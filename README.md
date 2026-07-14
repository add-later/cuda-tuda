# cuda-tuda
repo for 100 days of cuda challenge 

# Day 1
Kernel: `vecAdd.cu` 
- understanding grid, block and thread in CUDA
- read ch1 PMPP book 
- allocating and managing memory with `cudaMalloc`, `cudaMemcpy` and `cudaFree`

# Day 2
Kernel: `matAdd.cu`
- continued to getting familiar with grid, block and thread in CUDA
- solved https://leetgpu.com/challenges/matrix-addition with 1D block launching 
- read ch2 PMPP book

# Day 3
Kernel: `dotProduct.cu`
- solved https://www.deep-ml.com/projects/flash-attention-in-cuda-from-scratch/step/flash-attention-in-cuda-from-scratch-0006 with single thread
- read half of ch3 PMPP book

# Day 4
Kernel: `partSum.cu`
- solved https://leetgpu.com/challenges/reduction

# Day 5
Kernel `layerNorm.cu`
- wrote a layer normalization kernel

# Day 6
Kernel `matrixTranspose.cu`
- solved https://leetgpu.com/challenges/matrix-transpose
- read ch1-2 Deep Learning for CUDA

# Day 7
Kernel `matMul.cu`
- solved https://leetgpu.com/challenges/matrix-multiplication
- read PMPP book ch 4

# Day 8
Kernel `conv1D.cu`
- solved https://leetgpu.com/challenges/1d-convolution
- read PMPP book ch 7 (just the start)

# Day 9
Kernel `conv2D.cu`
- solved https://leetgpu.com/challenges/2d-convolution

# Day 10
Kernel `causalConv1D.cu`
- solved https://leetgpu.com/challenges/causal-depthwise-conv1d

# Day 11
Kernel `softmax.cu`
- solved https://leetgpu.com/challenges/softmax

# Day 12 
Kernel `swiglu.cu`
- solved https://leetgpu.com/challenges/swish-gated-linear-unit (I'm getting closer and closer for understanding what is going on! XD)

# Day 13
Kernel `prefixSum.cu`
- solved https://leetgpu.com/challenges/prefix-sum

# Day 14
Kernel `histogramKernel.cu`
- solved https://leetgpu.com/challenges/histogramming

# Day 15
Kernel `siLU.cu`
- solved https://leetgpu.com/challenges/sigmoid-linear-unit

# Day 16
Kernel `tiledMatmul.cu`
- wrote tiled matrix multiplication kernel
- started PMPP ch5 

# Day 17
Kernel `tiledTranspose.cu`
- wrote tiled matrix transposition kernel

# Day 18
Kernel `matrixCopy.cu`
- solved https://leetgpu.com/challenges/matrix-copy
- started watching lecture Compute and Memory Basics by GPU MODE

# Day 19 
Kernel `relu.cu`
- solved https://leetgpu.com/challenges/relu

# Day 20 
Kernel `cublasSgemm.cu`
- wrote matmul using cuBLAS and cuRAND

# Day 21
Kernel `rope.cu`
- solved https://leetgpu.com/challenges/rotary-positional-embedding

# Day 22
Kernel `rmsNorm.cu`
- solved https://leetgpu.com/challenges/rms-normalization

# Day 23
Kernel `rainbowTable.cu`
- solved https://leetgpu.com/challenges/rainbow-table

# Day 24
Kernel  `tiledMatmul_v2.cu`
- did a register tiling matmul kernel, but accepting only NxN matrices

# Day 25
Kernel `reluMatmulFused.cu`
- wrote a fused kernel, computing tiled matmul, adding bias and then relu

# Day 26 
Kernel `tiledMatmulGelu.cu`
- wrote fused matmul gelu kernel 

# Day 27
Kernel `clipKernel.cu`
- wrote value clipping kernel

# Day 28
Kernel `spmv.cu`
- wrote sparse matrix to dense vector multiplication kernel using hybrid ELL + COO

# Day 29
Kernel `streams.cu`
- wrote a kernel to persform simple transformation using CUDA streams

# Day 30 
Kernel `graph.cu`
- wrote a kernel scale -> add -> sqrt and launched using CUDA graph

# Day 31 
Kernel `reduceKernel.py`
- wrote a reduction kernel (https://leetgpu.com/challenges/reduction) and called it from python 

# Day 32 
Kernel `sigmoid.py`
- wrote a sigmoid activation kernel https://leetgpu.com/challenges/sigmoid-activation and profiled it with pycuda 

# Day 33
Kernel `interleave.cu`
- solved https://leetgpu.com/challenges/interleave-arrays

# Day 34
Kernel `geglu.cu`
- solved https://leetgpu.com/challenges/gaussian-error-gated-linear-unit

# Day 35
Kernel `reverse.cu`
- solved https://leetgpu.com/challenges/reverse-array

# Day 36
Kernel `wmma.cu`
- wrote fp16 matmul with WMMA API

# Day 37
Kernel `rgb.cu`
- solved https://leetgpu.com/challenges/rgb-to-grayscale

# Day 38
Kernel `bmm.cu`
- wrote batched matmul kernel with WMMA API

# Day 39 
Kernel `softmaxAttention.cu`
- solved https://leetgpu.com/challenges/softmax-attention

# Day 40 
Kernel `softmax.cu`
- upgraded my softmax kernel by making it O(N)

# Day 41
Kernel `tiledSoftmaxAttention.cu`
- added softmax to tiling computation

# Day 42
Kernel: `flashAttentionV1.ipynb`
- wrote flash attention v2 and called using torch

# Day 43
Kernel: `flashAttentionV2.ipynb`
- added causal masking and improved yesterday's code 

# Day 44
kernel: `geluStream.ipynb`
- wrote stream gelu kernel and compared it with pytorch