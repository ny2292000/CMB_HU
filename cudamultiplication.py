import pycuda.gpuarray as gpuarray
import numpy as np
import skcuda.linalg as linalg

# --- Initializations
import pycuda.autoinit
linalg.init()

A = np.array(([1, 2, 3], [4, 5, 6])).astype(np.float64)
B = np.array(([7, 8, 1, 5], [9, 10, 0, 9], [11, 12, 5, 5])).astype(np.float64)

A_gpu = gpuarray.to_gpu(A)
B_gpu = gpuarray.to_gpu(B)

C_gpu = linalg.dot(A_gpu, B_gpu)

print(np.dot(A, B))
print(C_gpu)