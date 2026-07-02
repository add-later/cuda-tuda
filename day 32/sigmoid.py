import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

mod = SourceModule("""
__global__ void sigmoid_kernel(const float* X, float* Y, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N)
        Y[idx] = 1 / (1 + expf(-X[idx]));
}
""")

N = 1 << 20 
X = np.random.randn(N).astype(np.float32)
Y = np.zeros(N, dtype=np.float32)

X_gpu = cuda.mem_alloc(X.nbytes)
Y_gpu = cuda.mem_alloc(Y.nbytes)
cuda.memcpy_htod(X_gpu, X)

kernel = mod.get_function("sigmoid_kernel")

start = cuda.Event()
stop  = cuda.Event()

start.record()
kernel(X_gpu, Y_gpu, np.int32(N),
       block=(256,1,1),
       grid=((N+255)//256, 1))
stop.record()
stop.synchronize()

print(f"Kernel time: {stop.time_since(start):.4f} ms")