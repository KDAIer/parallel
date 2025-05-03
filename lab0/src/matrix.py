import numpy as np
import time


def matrix(m, k, n):

    A = np.random.rand(m, k)
    B = np.random.rand(k, n)
    C = np.zeros((m, n))

    start = time.time()
    for i in range(m):
        for j in range(n):
            for l in range(k):
                C[i][j] += A[i][l] * B[l][j]
    end = time.time()
    elapsed = end - start
    print(f"Time: {elapsed:.6f}")

    flops = 2.0 * m * k * n
    gflops = flops / elapsed / 1e9
    print(f"浮点性能: {gflops:.6f}")

    return C


if __name__ == "__main__":
    matrix(512, 512, 512)
