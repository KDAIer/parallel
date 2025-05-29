// col_matmul.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void matMulCol(const float *A, const float *B, float *C,
                          int m, int n, int k) {
    // 这里我们把 threadIdx.x 用于行，threadIdx.y 用于列
    int row = blockIdx.y * blockDim.x + threadIdx.x;  
    int col = blockIdx.x * blockDim.y + threadIdx.y;  
    if (row < m && col < k) {
        float sum = 0.0f;
        for (int j = 0; j < n; ++j) {
            sum += A[row * n + j] * B[j * k + col];
        }
        C[row * k + col] = sum;
    }
}

int main(int argc, char **argv) {
    if (argc != 6) {
        printf("Usage: %s m n k block_x block_y\n", argv[0]);
        return -1;
    }
    int m = atoi(argv[1]), n = atoi(argv[2]), k = atoi(argv[3]);
    int bx = atoi(argv[4]), by = atoi(argv[5]);

    size_t sizeA = m * n * sizeof(float);
    size_t sizeB = n * k * sizeof(float);
    size_t sizeC = m * k * sizeof(float);

    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C = (float*)malloc(sizeC);
    for (int i = 0; i < m*n; ++i) h_A[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < n*k; ++i) h_B[i] = rand() / (float)RAND_MAX;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    dim3 block(bx, by);
    dim3 grid((k + bx - 1) / bx, (m + by - 1) / by);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matMulCol<<<grid, block>>>(d_A, d_B, d_C, m, n, k);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Column: %.3f ms\n", ms);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
