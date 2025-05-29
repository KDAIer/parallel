#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// 必须在编译时定义 TILE_DIM，比如 -DTILE_DIM=16
#ifndef TILE_DIM
#error "You must define TILE_DIM at compile time, e.g. nvcc -DTILE_DIM=16"
#endif

__global__ void matMulRowShared(const float *A, const float *B, float *C,
                                int m, int n, int k) {
    __shared__ float tileA[TILE_DIM][TILE_DIM];
    __shared__ float tileB[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (n + TILE_DIM - 1) / TILE_DIM; ++t) {
        int aCol = t * TILE_DIM + threadIdx.x;
        int bRow = t * TILE_DIM + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] = (row < m && aCol < n)
            ? A[row * n + aCol] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (bRow < n && col < k)
            ? B[bRow * k + col] : 0.0f;
        __syncthreads();

        for (int i = 0; i < TILE_DIM; ++i)
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        __syncthreads();
    }

    if (row < m && col < k)
        C[row * k + col] = sum;
}

int main(int argc, char **argv) {
    if (argc != 6) {
        printf("Usage: %s m n k block_x block_y\n", argv[0]);
        return -1;
    }
    int m = atoi(argv[1]), n = atoi(argv[2]), k = atoi(argv[3]);
    int bx = atoi(argv[4]), by = atoi(argv[5]);
    // bx,by must equal TILE_DIM

    size_t sizeA = m*n*sizeof(float), sizeB = n*k*sizeof(float), sizeC = m*k*sizeof(float);
    float *h_A = (float*)malloc(sizeA), *h_B = (float*)malloc(sizeB), *h_C = (float*)malloc(sizeC);
    for (int i=0;i<m*n;++i) h_A[i]=rand()/(float)RAND_MAX;
    for (int i=0;i<n*k;++i) h_B[i]=rand()/(float)RAND_MAX;

    float *d_A,*d_B,*d_C;
    cudaMalloc(&d_A,sizeA); cudaMalloc(&d_B,sizeB); cudaMalloc(&d_C,sizeC);
    cudaMemcpy(d_A,h_A,sizeA,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,sizeB,cudaMemcpyHostToDevice);

    dim3 block(bx,by), grid((k+TILE_DIM-1)/TILE_DIM,(m+TILE_DIM-1)/TILE_DIM);

    cudaEvent_t s,t; cudaEventCreate(&s); cudaEventCreate(&t);
    cudaEventRecord(s);
    matMulRowShared<<<grid,block>>>(d_A,d_B,d_C,m,n,k);
    cudaEventRecord(t); cudaEventSynchronize(t);

    float ms; cudaEventElapsedTime(&ms,s,t);
    printf("RowShared_%d: %.3f ms\n", TILE_DIM, ms);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
