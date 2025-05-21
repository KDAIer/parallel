#include<cstdio>
#include <cstdlib>       
#include <cuda_runtime.h> 

// CUDA kernel
__global__ void hello_world() {
    // 块索引
    int blockId = blockIdx.x;
    // 线程索引
    int threadId_x = threadIdx.x;
    int threadId_y = threadIdx.y;
    printf("Hello World from Thread (%d, %d) in Block %d!\n",
           threadId_x, threadId_y, blockId);
}


int main(int argc, char** argv) {
    if (argc != 4) {
        printf("Usage: %s n m k\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]);
    int m = atoi(argv[2]);
    int k = atoi(argv[3]);
    if (n < 1 || n > 32 || m < 1 || m > 32 || k < 1 || k > 32) {
        printf("n, m, k must be in [1, 32]\n");
        return 1;
    }
    dim3 blocks(n);
    dim3 threads(m, k);

    hello_world<<<blocks, threads>>>();
    // host 输出
    printf("Hello World from the host!\n");

    // 等待 GPU 完成
    cudaDeviceSynchronize();
    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    // 释放 GPU 资源
    cudaDeviceReset();
    return 0;
}