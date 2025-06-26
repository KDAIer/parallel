// im2col_opt.cu
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <cstring>

// 错误检查宏
#define CHECK_CUDA(call)                                                     \
    do                                                                       \
    {                                                                        \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess)                                              \
        {                                                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// ======================= matrix multiplication kernels =======================

// 基础全局内存矩阵乘法核函数 (二维线程块)
__global__ void matrixMulBasic(const float *A, const float *B, float *C, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < k)
    {
        float sum = 0.0f;
        for (int i = 0; i < n; i++)
        {
            sum += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = sum;
    }
}

// 共享内存优化矩阵乘法核 (二维线程块)，TILE_SIZE 由模板参数决定
template <int TILE_SIZE>
__global__ void matrixMulShared(const float *A, const float *B, float *C, int m, int n, int k)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    float sum = 0.0f;
    int numTiles = (n + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++)
    {
        int a_col = t * TILE_SIZE + tx;
        int b_row = t * TILE_SIZE + ty;
        if (row < m && a_col < n)
        {
            As[ty][tx] = A[row * n + a_col];
        }
        else
        {
            As[ty][tx] = 0.0f;
        }
        if (b_row < n && col < k)
        {
            Bs[ty][tx] = B[b_row * k + col];
        }
        else
        {
            Bs[ty][tx] = 0.0f;
        }
        __syncthreads();
        if (row < m && col < k)
        {
#pragma unroll
            for (int i = 0; i < TILE_SIZE; i++)
            {
                sum += As[ty][i] * Bs[i][tx];
            }
        }
        __syncthreads();
    }
    if (row < m && col < k)
    {
        C[row * k + col] = sum;
    }
}

// 寄存器优化矩阵乘法核 (二维线程块)，TILE_SIZE 由模板参数决定
template <int TILE_SIZE>
__global__ void matrixMulReged(const float *A, const float *B, float *C, int m, int n, int k)
{
    extern __shared__ float shared_buf[];
    // We'll split shared_buf into two TILE_SIZE x TILE_SIZE arrays
    float *tileA = shared_buf;
    float *tileB = shared_buf + TILE_SIZE * TILE_SIZE;
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    float sum = 0.0f;
    int numTiles = (n + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++)
    {
        int a_col = t * TILE_SIZE + tx;
        int b_row = t * TILE_SIZE + ty;
        // 载入到 shared
        if (row < m && a_col < n)
        {
            tileA[ty * TILE_SIZE + tx] = A[row * n + a_col];
        }
        else
        {
            tileA[ty * TILE_SIZE + tx] = 0.0f;
        }
        if (b_row < n && col < k)
        {
            tileB[ty * TILE_SIZE + tx] = B[b_row * k + col];
        }
        else
        {
            tileB[ty * TILE_SIZE + tx] = 0.0f;
        }
        __syncthreads();
        if (row < m && col < k)
        {
// 在寄存器中使用 tileA 和 tileB
#pragma unroll
            for (int i = 0; i < TILE_SIZE; i++)
            {
                float a_val = tileA[ty * TILE_SIZE + i];
                float b_val = tileB[i * TILE_SIZE + tx];
                sum += a_val * b_val;
            }
        }
        __syncthreads();
    }
    if (row < m && col < k)
    {
        C[row * k + col] = sum;
    }
}

// 1D分块版本 - 按行划分任务
__global__ void matrixMul1DRow(const float *A, const float *B, float *C, int m, int n, int k)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m)
    {
        for (int col = 0; col < k; col++)
        {
            float sum = 0.0f;
            for (int i = 0; i < n; i++)
            {
                sum += A[row * n + i] * B[i * k + col];
            }
            C[row * k + col] = sum;
        }
    }
}

// 1D分块版本 - 按列划分任务
__global__ void matrixMul1DCol(const float *A, const float *B, float *C, int m, int n, int k)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < k)
    {
        for (int row = 0; row < m; row++)
        {
            float sum = 0.0f;
            for (int i = 0; i < n; i++)
            {
                sum += A[row * n + i] * B[i * k + col];
            }
            C[row * k + col] = sum;
        }
    }
}

// im2col kernel：input NCHW with N=1, channels=3
__global__ void im2col_kernel(const float *input_data, int channels, int height, int width,
                              int kernel_h, int kernel_w, int stride_h, int stride_w,
                              int pad_h, int pad_w, int out_h, int out_w, float *col_data)
{
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_y < out_h && out_x < out_w)
    {
        int col_index = out_y * out_w + out_x;
        for (int c = 0; c < channels; ++c)
        {
            for (int kh = 0; kh < kernel_h; ++kh)
            {
                for (int kw = 0; kw < kernel_w; ++kw)
                {
                    int in_y = out_y * stride_h - pad_h + kh;
                    int in_x = out_x * stride_w - pad_w + kw;
                    int row_index = c * kernel_h * kernel_w + kh * kernel_w + kw;
                    int col_buffer_index = row_index * (out_h * out_w) + col_index;
                    if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width)
                    {
                        col_data[col_buffer_index] = input_data[c * height * width + in_y * width + in_x];
                    }
                    else
                    {
                        col_data[col_buffer_index] = 0.0f;
                    }
                }
            }
        }
    }
}

// 打印用法
void printUsage(const char *prog)
{
    std::cout << "Usage: " << prog << " <mode> <input_size> <block_size>\n";
    std::cout << "  mode: basic | shared | reged | row | col\n";
    std::cout << "  input_size: e.g., 32,256,512,... 矩阵宽高\n";
    std::cout << "  block_size: 线程块维度或 tile 大小 (e.g., 16,32)\n";
}

// 主函数
int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        printUsage(argv[0]);
        return 1;
    }
    const char *mode = argv[1];
    int input_size = atoi(argv[2]);
    int block_size = atoi(argv[3]);
    if (input_size <= 0 || block_size <= 0)
    {
        std::cerr << "Invalid input_size or block_size\n";
        return 1;
    }
    // parameters
    const int in_channels = 3;
    const int out_channels = 3;
    const int kernel_h = 3, kernel_w = 3;
    const int stride = 1;
    const int pad = (kernel_h - 1) / 2; // same padding

    int in_h = input_size, in_w = input_size;
    int out_h = (in_h + 2 * pad - kernel_h) / stride + 1;
    int out_w = (in_w + 2 * pad - kernel_w) / stride + 1;

    // Allocate host input & kernel (随机初始化)
    size_t in_elems = in_channels * in_h * in_w;
    size_t kernel_elems = out_channels * in_channels * kernel_h * kernel_w;
    std::vector<float> h_input(in_elems);
    std::vector<float> h_kernel(kernel_elems);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (size_t i = 0; i < in_elems; i++)
        h_input[i] = dis(gen);
    for (size_t i = 0; i < kernel_elems; i++)
        h_kernel[i] = dis(gen);

    // Allocate device memory
    float *d_input = nullptr, *d_kernel = nullptr, *d_col = nullptr, *d_output = nullptr;
    CHECK_CUDA(cudaMalloc(&d_input, in_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_kernel, kernel_elems * sizeof(float)));
    size_t col_buffer_size = (size_t)in_channels * kernel_h * kernel_w * out_h * out_w;
    CHECK_CUDA(cudaMalloc(&d_col, col_buffer_size * sizeof(float)));
    size_t out_elems = (size_t)out_channels * out_h * out_w;
    CHECK_CUDA(cudaMalloc(&d_output, out_elems * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), in_elems * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kernel, h_kernel.data(), kernel_elems * sizeof(float), cudaMemcpyHostToDevice));

    // Prepare im2col grid
    dim3 block2d(16, 16);
    dim3 grid2d((out_w + block2d.x - 1) / block2d.x,
                (out_h + block2d.y - 1) / block2d.y);
    // We use a fixed 16x16 for im2col; performance impact small compared to GEMM.

    // Launch im2col once (no need to warm repeatedly here)
    CHECK_CUDA(cudaMemset(d_col, 0, col_buffer_size * sizeof(float)));
    im2col_kernel<<<grid2d, block2d>>>(d_input, in_channels, in_h, in_w,
                                       kernel_h, kernel_w, stride, stride,
                                       pad, pad, out_h, out_w, d_col);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Prepare GEMM sizes: A = kernel matrix (m x n), B = col matrix (n x k), C = output (m x k)
    int m = out_channels;                      // 通常 3
    int n = in_channels * kernel_h * kernel_w; // 3*3*3=27
    int k = out_h * out_w;                     // 例如 256*256

    // Copy kernel into a properly shaped matrix A_dev: we can reuse d_kernel with index arithmetic
    // But for simplicity we use d_kernel as A in row-major: A[row=i_out][col=j] where
    // row index: out channel, col index: flattened in channel*kernel_h*kernel_w
    // B is d_col already in shape (n x k)
    // C output in shape (m x k) stored in d_output.

    // Setup timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Launch chosen GEMM kernel with given block_size
    // We measure only the GEMM part, but since im2col is once, measure from GEMM start to end.
    float elapsed_ms = 0.0f;
    // Warmup: optionally run once before timing
    {
        if (strcmp(mode, "basic") == 0)
        {
            dim3 grid((k + block_size - 1) / block_size,
                      (m + block_size - 1) / block_size);
            dim3 blk(block_size, block_size);
            matrixMulBasic<<<grid, blk>>>(d_kernel, d_col, d_output, m, n, k);
        }
        else if (strcmp(mode, "shared") == 0)
        {
            // template requires compile-time TILE_SIZE; but block_size known at runtime.
            // We handle common TILE sizes: 16, 32, 64. For others, fallback to basic.
            if (block_size == 16)
            {
                dim3 grid((k + 16 - 1) / 16, (m + 16 - 1) / 16);
                dim3 blk(16, 16);
                matrixMulShared<16><<<grid, blk>>>(d_kernel, d_col, d_output, m, n, k);
            }
            else if (block_size == 32)
            {
                dim3 grid((k + 32 - 1) / 32, (m + 32 - 1) / 32);
                dim3 blk(32, 32);
                matrixMulShared<32><<<grid, blk>>>(d_kernel, d_col, d_output, m, n, k);
            }
            else if (block_size == 64)
            {
                dim3 grid((k + 64 - 1) / 64, (m + 64 - 1) / 64);
                dim3 blk(64, 64);
                matrixMulShared<64><<<grid, blk>>>(d_kernel, d_col, d_output, m, n, k);
            }
            else
            {
                // 不支持的 tile 大小，fallback 基础
                dim3 grid((k + block_size - 1) / block_size, (m + block_size - 1) / block_size);
                dim3 blk(block_size, block_size);
                matrixMulBasic<<<grid, blk>>>(d_kernel, d_col, d_output, m, n, k);
            }
        }
        else if (strcmp(mode, "reged") == 0)
        {
            if (block_size == 16)
            {
                dim3 grid((k + 16 - 1) / 16, (m + 16 - 1) / 16);
                dim3 blk(16, 16);
                size_t shared_sz = 2 * 16 * 16 * sizeof(float);
                matrixMulReged<16><<<grid, blk, shared_sz>>>(d_kernel, d_col, d_output, m, n, k);
            }
            else if (block_size == 32)
            {
                dim3 grid((k + 32 - 1) / 32, (m + 32 - 1) / 32);
                dim3 blk(32, 32);
                size_t shared_sz = 2 * 32 * 32 * sizeof(float);
                matrixMulReged<32><<<grid, blk, shared_sz>>>(d_kernel, d_col, d_output, m, n, k);
            }
            else
            {
                // 不支持其他大 tile，fallback 基础
                dim3 grid((k + block_size - 1) / block_size, (m + block_size - 1) / block_size);
                dim3 blk(block_size, block_size);
                matrixMulBasic<<<grid, blk>>>(d_kernel, d_col, d_output, m, n, k);
            }
        }
        // else if (strcmp(mode, "row") == 0)
        // {
        //     // 1D 按行：m 行，k 列
        //     int threads = block_size;
        //     int blocks = (m + threads - 1) / threads;
        //     matrixMul1DRow<<<blocks, threads>>>(d_kernel, d_col, d_output, m, n, k);
        // }
        // else if (strcmp(mode, "col") == 0)
        // {
        //     // 1D 按列
        //     int threads = block_size;
        //     int blocks = (k + threads - 1) / threads;
        //     matrixMul1DCol<<<blocks, threads>>>(d_kernel, d_col, d_output, m, n, k);
        // }
        else
        {
            std::cerr << "Unknown mode: " << mode << "\n";
            return 1;
        }
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // Timing
    CHECK_CUDA(cudaEventRecord(start));
    if (strcmp(mode, "basic") == 0)
    {
        dim3 grid((k + block_size - 1) / block_size,
                  (m + block_size - 1) / block_size);
        dim3 blk(block_size, block_size);
        matrixMulBasic<<<grid, blk>>>(d_kernel, d_col, d_output, m, n, k);
    }
    else if (strcmp(mode, "shared") == 0)
    {
        if (block_size == 8)
        {
            dim3 grid((k + 8 - 1) / 8, (m + 8 - 1) / 8);
            dim3 blk(8, 8);
            matrixMulShared<8><<<grid, blk>>>(d_kernel, d_col, d_output, m, n, k);
        }
        if (block_size == 16)
        {
            dim3 grid((k + 16 - 1) / 16, (m + 16 - 1) / 16);
            dim3 blk(16, 16);
            matrixMulShared<16><<<grid, blk>>>(d_kernel, d_col, d_output, m, n, k);
        }
        else if (block_size == 32)
        {
            dim3 grid((k + 32 - 1) / 32, (m + 32 - 1) / 32);
            dim3 blk(32, 32);
            matrixMulShared<32><<<grid, blk>>>(d_kernel, d_col, d_output, m, n, k);
        }
        else if (block_size == 64)
        {
            dim3 grid((k + 64 - 1) / 64, (m + 64 - 1) / 64);
            dim3 blk(64, 64);
            matrixMulShared<64><<<grid, blk>>>(d_kernel, d_col, d_output, m, n, k);
        }
        else
        {
            dim3 grid((k + block_size - 1) / block_size, (m + block_size - 1) / block_size);
            dim3 blk(block_size, block_size);
            matrixMulBasic<<<grid, blk>>>(d_kernel, d_col, d_output, m, n, k);
        }
    }
    else if (strcmp(mode, "reged") == 0)
    {
        if (block_size == 16)
        {
            dim3 grid((k + 16 - 1) / 16, (m + 16 - 1) / 16);
            dim3 blk(16, 16);
            size_t shared_sz = 2 * 16 * 16 * sizeof(float);
            matrixMulReged<16><<<grid, blk, shared_sz>>>(d_kernel, d_col, d_output, m, n, k);
        }
        else if (block_size == 32)
        {
            dim3 grid((k + 32 - 1) / 32, (m + 32 - 1) / 32);
            dim3 blk(32, 32);
            size_t shared_sz = 2 * 32 * 32 * sizeof(float);
            matrixMulReged<32><<<grid, blk, shared_sz>>>(d_kernel, d_col, d_output, m, n, k);
        }
        else
        {
            dim3 grid((k + block_size - 1) / block_size, (m + block_size - 1) / block_size);
            dim3 blk(block_size, block_size);
            matrixMulBasic<<<grid, blk>>>(d_kernel, d_col, d_output, m, n, k);
        }
    }
    // else if (strcmp(mode, "row") == 0)
    // {
    //     int threads = block_size;
    //     int blocks = (m + threads - 1) / threads;
    //     matrixMul1DRow<<<blocks, threads>>>(d_kernel, d_col, d_output, m, n, k);
    // }
    // else if (strcmp(mode, "col") == 0)
    // {
    //     int threads = block_size;
    //     int blocks = (k + threads - 1) / threads;
    //     matrixMul1DCol<<<blocks, threads>>>(d_kernel, d_col, d_output, m, n, k);
    // }
    // 同步并测时
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // Copy back if needed (此处不检查结果，仅测时)
    // CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, out_elems*sizeof(float), cudaMemcpyDeviceToHost));

    // 打印结果行，供脚本提取
    printf("Mode: %s, Size: %d, Block: %d, Time: %.6f ms\n",
           mode, input_size, block_size, elapsed_ms);

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_col);
    cudaFree(d_output);

    return 0;
}
