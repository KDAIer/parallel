// im2col_opt.cu
#include <cstdio>
#include <vector>
#include <random>
#include <cuda_runtime.h>

// 错误检查宏
#define CHECK_CUDA(call)                                                     \
    do                                                                       \
    {                                                                        \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess)                                              \
        {                                                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// ==================== 矩阵乘法核函数 ====================
// 行划分 (1D)：每个线程处理一整行
__global__ void matrixMulRow(const float *A, const float *B, float *C, int m, int n, int k)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m)
    {
        for (int col = 0; col < k; ++col)
        {
            float sum = 0;
            for (int i = 0; i < n; ++i)
            {
                sum += A[row * n + i] * B[i * k + col];
            }
            C[row * k + col] = sum;
        }
    }
}

// 列划分 (1D)：每个线程处理一整列
__global__ void matrixMulCol(const float *A, const float *B, float *C, int m, int n, int k)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < k)
    {
        for (int row = 0; row < m; ++row)
        {
            float sum = 0;
            for (int i = 0; i < n; ++i)
            {
                sum += A[row * n + i] * B[i * k + col];
            }
            C[row * k + col] = sum;
        }
    }
}

// 二维块划分 (2D)：每个线程处理一个元素
__global__ void matrixMulBlock(const float *A, const float *B, float *C, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < k)
    {
        float sum = 0;
        for (int i = 0; i < n; ++i)
        {
            sum += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = sum;
    }
}

// 共享内存优化 (示例 TILE_SIZE=16)
template <int TILE_SIZE>
__global__ void matrixMulShared(const float *A, const float *B, float *C, int m, int n, int k)
{
    __shared__ float sA[TILE_SIZE][TILE_SIZE], sB[TILE_SIZE][TILE_SIZE];
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    float sum = 0;
    int numTiles = (n + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; ++t)
    {
        int aCol = t * TILE_SIZE + tx;
        int bRow = t * TILE_SIZE + ty;
        sA[ty][tx] = (row < m && aCol < n) ? A[row * n + aCol] : 0.0f;
        sB[ty][tx] = (bRow < n && col < k) ? B[bRow * k + col] : 0.0f;
        __syncthreads();
        if (row < m && col < k)
        {
#pragma unroll
            for (int i = 0; i < TILE_SIZE; ++i)
                sum += sA[ty][i] * sB[i][tx];
        }
        __syncthreads();
    }
    if (row < m && col < k)
        C[row * k + col] = sum;
}

// im2col kernel
__global__ void im2col_kernel(const float *in, int C, int H, int W,
                              int KH, int KW, int S, int P,
                              int outH, int outW, float *col)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y < outH && x < outW)
    {
        int idx = y * outW + x;
        for (int c = 0; c < C; ++c)
            for (int ky = 0; ky < KH; ++ky)
                for (int kx = 0; kx < KW; ++kx)
                {
                    int inY = y * S - P + ky;
                    int inX = x * S - P + kx;
                    int row = c * KH * KW + ky * KW + kx;
                    if (inY >= 0 && inY < H && inX >= 0 && inX < W)
                        col[row * (outH * outW) + idx] = in[c * H * W + inY * W + inX];
                    else
                        col[row * (outH * outW) + idx] = 0.0f;
                }
    }
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("Usage: %s <input_size> <block_size>\n", argv[0]);
        return 1;
    }
    int Nsize = atoi(argv[1]);
    int Bsz = atoi(argv[2]);
    // parameters
    const int C = 3, Kc = 3, S = 1, P = 1; // kernel=3, stride=1, pad=1
    int H = Nsize, W = Nsize;
    int outH = (H + 2 * P - Kc) / S + 1,
        outW = (W + 2 * P - Kc) / S + 1;
    // im2col dims
    int m = 3, n = C * Kc * Kc, k = outH * outW;
    // alloc host
    std::vector<float> h_in(C * H * W), h_ker(C * Kc * Kc);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1, 1);
    for (auto &v : h_in)
        v = dis(gen);
    for (auto &v : h_ker)
        v = dis(gen);
    // alloc dev
    float *d_in, *d_ker, *d_col, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, h_in.size() * 4));
    CHECK_CUDA(cudaMalloc(&d_ker, h_ker.size() * 4));
    CHECK_CUDA(cudaMalloc(&d_col, n * k * 4));
    CHECK_CUDA(cudaMalloc(&d_out, m * k * 4));
    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), h_in.size() * 4, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ker, h_ker.data(), h_ker.size() * 4, cudaMemcpyHostToDevice));
    // im2col
    dim3 b2(16, 16), g2((outW + 15) / 16, (outH + 15) / 16);
    im2col_kernel<<<g2, b2>>>(d_in, C, H, W, Kc, Kc, S, P, outH, outW, d_col);
    CHECK_CUDA(cudaDeviceSynchronize());
    // timing
    cudaEvent_t st, ed;
    CHECK_CUDA(cudaEventCreate(&st));
    CHECK_CUDA(cudaEventCreate(&ed));
    float t;
    // 1) 行划分
    CHECK_CUDA(cudaEventRecord(st));
    matrixMulRow<<<(m + Bsz - 1) / Bsz, Bsz>>>(d_ker, d_col, d_out, m, n, k);
    CHECK_CUDA(cudaEventRecord(ed));
    CHECK_CUDA(cudaEventSynchronize(ed));
    CHECK_CUDA(cudaEventElapsedTime(&t, st, ed));
    printf("Mode: row,  Size: %d, Block: %d, Time: %.6f ms\n", Nsize, Bsz, t);
    // 2) 列划分
    CHECK_CUDA(cudaEventRecord(st));
    matrixMulCol<<<(k + Bsz - 1) / Bsz, Bsz>>>(d_ker, d_col, d_out, m, n, k);
    CHECK_CUDA(cudaEventRecord(ed));
    CHECK_CUDA(cudaEventSynchronize(ed));
    CHECK_CUDA(cudaEventElapsedTime(&t, st, ed));
    printf("Mode: col,  Size: %d, Block: %d, Time: %.6f ms\n", Nsize, Bsz, t);
    // 3) 二维块划分
    dim3 b22(Bsz, Bsz), g22((k + Bsz - 1) / Bsz, (m + Bsz - 1) / Bsz);
    CHECK_CUDA(cudaEventRecord(st));
    matrixMulBlock<<<g22, b22>>>(d_ker, d_col, d_out, m, n, k);
    CHECK_CUDA(cudaEventRecord(ed));
    CHECK_CUDA(cudaEventSynchronize(ed));
    CHECK_CUDA(cudaEventElapsedTime(&t, st, ed));
    printf("Mode: block,Size: %d, Block: %d, Time: %.6f ms\n", Nsize, Bsz, t);
    // 4) 共享内存划分 (示例 TILE=16)
    // if (Bsz == 16)
    // {
    //     dim3 g3((k + 15) / 16, (m + 15) / 16), b3(16, 16);
    //     CHECK_CUDA(cudaEventRecord(st));
    //     matrixMulShared<16><<<g3, b3>>>(d_ker, d_col, d_out, m, n, k);
    //     CHECK_CUDA(cudaEventRecord(ed));
    //     CHECK_CUDA(cudaEventSynchronize(ed));
    //     CHECK_CUDA(cudaEventElapsedTime(&t, st, ed));
    //     printf("Mode: shared,Size: %d, Block: %d, Time: %.6f ms\n", Nsize, Bsz, t);
    // }
    // cleanup
    cudaFree(d_in);
    cudaFree(d_ker);
    cudaFree(d_col);
    cudaFree(d_out);
    return 0;
}
