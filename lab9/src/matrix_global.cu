#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void transpose(float* out, float* in, int n) {
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    out[(bx + tx) * n + by + ty] = in[(by + ty) * n + bx + tx];
}

void random_init(float* data, int n) {
    for (int i = 0; i < n * n; i++)
        data[i] = static_cast<float>(rand()) / RAND_MAX;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: %s n B\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]);
    int B = atoi(argv[2]);
    if (n < 512 || n > 2048) {
        printf("n must be in [512,2048]\n");
        return 1;
    }
    size_t bytes = n * n * sizeof(float);

    // host_A 用来存储输入矩阵，host_B 用来存储输出矩阵
    float *host_A = (float*)malloc(bytes), *host_B = (float*)malloc(bytes);
    random_init(host_A, n);

    // device_A 用来存储输入矩阵，device_B 用来存储输出矩阵
    float *device_A, *device_B;

    cudaMalloc(&device_A, bytes);
    cudaMalloc(&device_B, bytes);
    // 把 host_A 复制到 device_A
    cudaMemcpy(device_A, host_A, bytes, cudaMemcpyHostToDevice);

    dim3 block(B, B);
    dim3 grid((n + B - 1) / B, (n + B - 1) / B);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    transpose<<<grid, block>>>(device_A, device_B, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // 把 device_B 复制到 host_B
    cudaMemcpy(host_B, device_B, bytes, cudaMemcpyDeviceToHost);
    printf("Global transpose time: %.3f ms\n", ms);

    cudaFree(device_A);
    cudaFree(device_B);
    free(host_A);
    free(host_B);
    return 0;
}