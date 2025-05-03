#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>
#include <time.h>

// 共享信息结构体
typedef struct
{
    int m, n, k; // A为m×n，B为n×k，C为m×k
    double *A, *B, *C;
} share_info;

// 每个线程的私有信息
typedef struct
{
    share_info *info;
    int start_row;
    int end_row;
} thread_info;

void *thread_func(void *arg)
{
    thread_info *tinfo = (thread_info *)arg;
    share_info *info = tinfo->info;
    int m = info->m;
    int n = info->n;
    int k = info->k;
    int start = tinfo->start_row;
    int end = tinfo->end_row;

    for (int i = start; i < end; i++)
    {
        for (int j = 0; j < k; j++)
        {
            double sum = 0.0;
            for (int p = 0; p < n; p++)
            {
                sum += info->A[i * n + p] * info->B[p * k + j];
            }
            info->C[i * k + j] = sum;
        }
    }
    return NULL;
}

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        fprintf(stderr, "用法: %s <m> <n> <k> <num_threads>\n", argv[0]);
        return 1;
    }
    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);
    int num_threads = atoi(argv[4]);

    // 分配共享结构体与矩阵空间
    share_info info;
    info.m = m;
    info.n = n;
    info.k = k;
    info.A = (double *)malloc(m * n * sizeof(double));
    info.B = (double *)malloc(n * k * sizeof(double));
    info.C = (double *)malloc(m * k * sizeof(double));
    if (!info.A || !info.B || !info.C)
    {
        fprintf(stderr, "内存分配失败！\n");
        return 1;
    }

    // 使用系统时间作为随机数种子
    srand((unsigned int)time(NULL));
    // 随机初始化矩阵A和B
    for (int i = 0; i < m * n; i++)
    {
        info.A[i] = (double)(rand() % 100) / 10.0;
    }
    for (int i = 0; i < n * k; i++)
    {
        info.B[i] = (double)(rand() % 100) / 10.0;
    }

    // 创建线程以及设置每个线程所计算的矩阵行区间
    pthread_t threads[num_threads];
    thread_info tinfo[num_threads];
    int rows_per_thread = m / num_threads;
    int extra = m % num_threads;
    int current_row = 0;

    // 记录开始时间
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    for (int i = 0; i < num_threads; i++)
    {
        tinfo[i].info = &info;
        tinfo[i].start_row = current_row;
        // 如果还有剩余行则均匀分配
        tinfo[i].end_row = current_row + rows_per_thread + (i < extra ? 1 : 0);
        current_row = tinfo[i].end_row;
        pthread_create(&threads[i], NULL, thread_func, (void *)&tinfo[i]);
    }

    // 等待所有线程完成工作
    for (int i = 0; i < num_threads; i++)
    {
        pthread_join(threads[i], NULL);
    }

    // 记录结束时间
    gettimeofday(&end_time, NULL);
    double time_taken = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec) / 1e6;

    printf("矩阵相乘耗时: %.5f s\n", time_taken);

    // 释放内存
    free(info.A);
    free(info.B);
    free(info.C);

    return 0;
}
