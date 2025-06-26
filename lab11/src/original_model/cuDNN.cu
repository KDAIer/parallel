#include <iostream>
#include <vector>
#include <random>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cudnn.h>

// 错误检查宏
#define CHECK_CUDA(call)                                                 \
    do                                                                   \
    {                                                                    \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess)                                          \
        {                                                                \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl;  \
            std::exit(EXIT_FAILURE);                                     \
        }                                                                \
    } while (0)

#define CHECK_CUDNN(call)                                                   \
    do                                                                      \
    {                                                                       \
        cudnnStatus_t status = call;                                        \
        if (status != CUDNN_STATUS_SUCCESS)                                 \
        {                                                                   \
            std::cerr << "CUDNN error at " << __FILE__ << ":" << __LINE__   \
                      << " - " << cudnnGetErrorString(status) << std::endl; \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

int main(int argc, char *argv[])
{
    if (argc < 3 || argc > 4)
    {
        std::cerr << "Usage: " << argv[0] << " <input_size> <stride> [batch_size]\n";
        std::cerr << "  input_size: e.g., 32, 256, 512, ...\n";
        std::cerr << "  stride: 1, 2, or 3\n";
        std::cerr << "  batch_size: optional, default 1\n";
        return EXIT_FAILURE;
    }
    int input_size = std::atoi(argv[1]);
    int stride = std::atoi(argv[2]);
    int batch = 1;
    if (argc == 4)
    {
        batch = std::atoi(argv[3]);
    }
    if (input_size <= 0 || stride <= 0 || batch <= 0)
    {
        std::cerr << "Invalid arguments: input_size, stride, batch_size must be positive integers.\n";
        return EXIT_FAILURE;
    }

    // cuDNN handle
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    // 参数设置
    const int N = batch;
    const int C = 3;
    const int H = input_size;
    const int W = input_size;
    const int K = 3; // 输出通道数
    const int FH = 3, FW = 3;
    const int SH = stride, SW = stride;
    // same padding: pad = (kernel_size - 1) / 2
    const int pad = (FH - 1) / 2; // =1 for FH=3

    // 创建描述符
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_desc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));

    // 设置 input tensor 描述 (N, C, H, W)
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));

    // 设置 filter 描述 (K, C, FH, FW)
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(
        filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, K, C, FH, FW));

    // 设置 convolution 描述 (pad, stride, dilation=1)
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
        conv_desc, pad, pad, SH, SW, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // 计算输出尺寸并设置 output tensor 描述
    int out_N, out_C, out_H, out_W;
    CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(
        conv_desc, input_desc, filter_desc, &out_N, &out_C, &out_H, &out_W));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_N, out_C, out_H, out_W));

    std::cout << "cuDNN Benchmark: Input " << H << "x" << W
              << ", Channels=" << C << ", Output Channels=" << K
              << ", Stride=" << SH << ", Batch=" << N << "\n";
    std::cout << "Output Size: " << out_H << "x" << out_W << "\n";

    // 分配并初始化 host/device 内存
    size_t in_bytes = (size_t)N * C * H * W * sizeof(float);
    size_t filter_bytes = (size_t)K * C * FH * FW * sizeof(float);
    size_t out_bytes = (size_t)out_N * out_C * out_H * out_W * sizeof(float);

    std::vector<float> h_input(N * C * H * W);
    std::vector<float> h_filter(K * C * FH * FW);
    // 随机初始化
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (auto &v : h_input)
        v = dis(gen);
    for (auto &v : h_filter)
        v = dis(gen);

    float *d_input = nullptr, *d_filter = nullptr, *d_output = nullptr;
    CHECK_CUDA(cudaMalloc(&d_input, in_bytes));
    CHECK_CUDA(cudaMalloc(&d_filter, filter_bytes));
    CHECK_CUDA(cudaMalloc(&d_output, out_bytes));

    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), in_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_filter, h_filter.data(), filter_bytes, cudaMemcpyHostToDevice));

    // 选算法：使用 cudnnGetConvolutionForwardAlgorithm_v7 枚举候选并测时选择最优
    int max_algos = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
    std::vector<cudnnConvolutionFwdAlgoPerf_t> perfResults(max_algos);
    int returnedAlgoCount = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(
        cudnn, input_desc, filter_desc, conv_desc, output_desc,
        max_algos, &returnedAlgoCount, perfResults.data()));
    std::cout << "Found " << returnedAlgoCount << " convolution algorithms\n";

    float bestTime = 1e20f;
    cudnnConvolutionFwdAlgo_t bestAlgo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    size_t bestWs = 0;

    // 遍历候选算法，测时并选择
    for (int i = 0; i < returnedAlgoCount; ++i)
    {
        auto &p = perfResults[i];
        if (p.status != CUDNN_STATUS_SUCCESS)
            continue;
        cudnnConvolutionFwdAlgo_t algo = p.algo;
        size_t ws_bytes = 0;
        CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
            cudnn, input_desc, filter_desc, conv_desc, output_desc, algo, &ws_bytes));
        // 可根据显存情况设阈值，例如 500MB
        const size_t MAX_WS = (size_t)500 << 20;
        if (ws_bytes > MAX_WS)
        {
            std::cout << "Skip algo " << algo << " due to large workspace "
                      << (ws_bytes >> 20) << " MB\n";
            continue;
        }
        void *d_ws = nullptr;
        if (ws_bytes > 0)
        {
            CHECK_CUDA(cudaMalloc(&d_ws, ws_bytes));
        }
        // Warmup
        float alpha = 1.0f, beta = 0.0f;
        const int WARMUP_ITERS = 3;
        for (int t = 0; t < WARMUP_ITERS; ++t)
        {
            CHECK_CUDNN(cudnnConvolutionForward(
                cudnn, &alpha, input_desc, d_input,
                filter_desc, d_filter, conv_desc,
                algo, d_ws, ws_bytes, &beta, output_desc, d_output));
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        // 多次测时取平均
        const int ITERS = 5;
        cudaEvent_t startEvt, stopEvt;
        CHECK_CUDA(cudaEventCreate(&startEvt));
        CHECK_CUDA(cudaEventCreate(&stopEvt));
        float totalTime = 0.0f;
        for (int t = 0; t < ITERS; ++t)
        {
            CHECK_CUDA(cudaEventRecord(startEvt));
            CHECK_CUDNN(cudnnConvolutionForward(
                cudnn, &alpha, input_desc, d_input,
                filter_desc, d_filter, conv_desc,
                algo, d_ws, ws_bytes, &beta, output_desc, d_output));
            CHECK_CUDA(cudaEventRecord(stopEvt));
            CHECK_CUDA(cudaEventSynchronize(stopEvt));
            float ms = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&ms, startEvt, stopEvt));
            totalTime += ms;
        }
        float avgTime = totalTime / ITERS;
        std::cout << "Algo " << algo << " avg time: " << avgTime
                  << " ms, workspace: " << (ws_bytes >> 20) << " MB\n";
        if (avgTime < bestTime)
        {
            bestTime = avgTime;
            bestAlgo = algo;
            bestWs = ws_bytes;
        }
        CHECK_CUDA(cudaEventDestroy(startEvt));
        CHECK_CUDA(cudaEventDestroy(stopEvt));
        if (d_ws)
            cudaFree(d_ws);
    }

    std::cout << "Best algo: " << bestAlgo
              << ", avg time: " << bestTime << " ms, workspace: "
              << (bestWs >> 20) << " MB\n";

    // 最终运行一次并输出 “Execution time: X ms”
    void *d_best_ws = nullptr;
    if (bestWs > 0)
    {
        CHECK_CUDA(cudaMalloc(&d_best_ws, bestWs));
    }
    // Warmup final
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUDNN(cudnnConvolutionForward(
        cudnn, &alpha, input_desc, d_input,
        filter_desc, d_filter, conv_desc,
        bestAlgo, d_best_ws, bestWs, &beta, output_desc, d_output));
    CHECK_CUDA(cudaDeviceSynchronize());
    // 计时 final
    cudaEvent_t finalStart, finalStop;
    CHECK_CUDA(cudaEventCreate(&finalStart));
    CHECK_CUDA(cudaEventCreate(&finalStop));
    CHECK_CUDA(cudaEventRecord(finalStart));
    CHECK_CUDNN(cudnnConvolutionForward(
        cudnn, &alpha, input_desc, d_input,
        filter_desc, d_filter, conv_desc,
        bestAlgo, d_best_ws, bestWs, &beta, output_desc, d_output));
    CHECK_CUDA(cudaEventRecord(finalStop));
    CHECK_CUDA(cudaEventSynchronize(finalStop));
    float final_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&final_ms, finalStart, finalStop));
    std::cout << "Execution time: " << final_ms << " ms\n";

    // Cleanup
    if (d_best_ws)
        cudaFree(d_best_ws);
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);

    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn);

    CHECK_CUDA(cudaEventDestroy(finalStart));
    CHECK_CUDA(cudaEventDestroy(finalStop));

    return 0;
}
