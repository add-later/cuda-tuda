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