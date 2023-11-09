import cupy as cp
import numpy as np
import time

# 生成随机矩阵
N = 1000
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)

# 将矩阵转换为Cupy数组
A_gpu = cp.asarray(A)
B_gpu = cp.asarray(B)

# CPU计算矩阵乘法
start = time.time()
C_cpu = np.dot(A, B)
cpu_time = time.time() - start