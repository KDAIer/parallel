#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define BDIM 32   // 支持的最大线程块边长

__global__ void transpose_shared(float* out, float* in, int n){
    __shared__ float smem[BDIM*BDIM];
    int B = blockDim.x;               // 实际块大小
    int bx = blockIdx.x * B;
    int by = blockIdx.y * B;
    int tx = threadIdx.x, ty = threadIdx.y;
    int x = bx + tx, y = by + ty;
    if (x < n && y < n) {
        smem[ty*BDIM + tx] = in[y*n + x];
    }
    __syncthreads();
    // 转置写回
    int x2 = by + tx, y2 = bx + ty;
    if (x2 < n && y2 < n) {
        out[y2*n + x2] = smem[tx*BDIM + ty];
    }
}

static void random_init(float* a, int n){
    for(int i=0;i<n*n;i++)
        a[i] = rand() / (float)RAND_MAX;
}

int main(int argc, char** argv){
    if(argc!=3){
        printf("Usage: %s <N> <B>\n", argv[0]);
        return 1;
    }
    int N = atoi(argv[1]);
    int B = atoi(argv[2]);
    if (N<512 || N>2048 || B<1 || B>BDIM){
        printf("Require: 512<=N<=2048, 1<=B<=%d\n", BDIM);
        return 1;
    }

    size_t bytes = N*(size_t)N * sizeof(float);
    float *h_A = (float*)malloc(bytes),
          *h_B = (float*)malloc(bytes);
    random_init(h_A, N);

    float *d_A, *d_B;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);

    dim3 block(B, B),
         grid( (N+B-1)/B, (N+B-1)/B );

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    transpose_shared<<<grid, block>>>(d_B, d_A, N);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost);
    printf("Shared transpose time: %.3f ms\n", ms);

    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);
    return 0;
}