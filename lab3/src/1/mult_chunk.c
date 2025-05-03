#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>

// 共享信息结构体：A 为 m×n，B 为 n×k，C 为 m×k
typedef struct
{
    int m, n, k;
    double *A, *B, *C;
} share_info;

// 每个线程的私有信息（用于2D块划分）
typedef struct
{
    share_info *info;
    int row_start, row_end; // 输出矩阵 C 的行区间 [row_start, row_end)
    int col_start, col_end; // 输出矩阵 C 的列区间 [col_start, col_end)
} block_thread_info;

// 每个线程的工作函数，计算子块 C(row_start:row_end, col_start:col_end)
void *block_thread_func(void *arg)
{
    block_thread_info *btinfo = (block_thread_info *)arg;
    share_info *info = btinfo->info;
    int m = info->m, n = info->n, k = info->k;
    int rs = btinfo->row_start, re = btinfo->row_end;
    int cs = btinfo->col_start, ce = btinfo->col_end;
    for (int i = rs; i < re; i++)
    {
        for (int j = cs; j < ce; j++)
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

    // 分配共享结构体及矩阵空间
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

    // 使用系统时间作为随机数种子，随机初始化矩阵 A 和 B
    srand((unsigned int)time(NULL));
    for (int i = 0; i < m * n; i++)
    {
        info.A[i] = (double)(rand() % 100) / 10.0;
    }
    for (int i = 0; i < n * k; i++)
    {
        info.B[i] = (double)(rand() % 100) / 10.0;
    }

    // 计算二维分块数，行方向 R = floor(sqrt(num_threads))
    int R = (int)floor(sqrt((double)num_threads));
    if (R < 1)
        R = 1;
    int C = (num_threads + R - 1) / R; // 列方向块数
    // 每一块的行数和余数
    int base_row = m / R;
    int extra_row = m % R;
    // 每一块的列数和余数
    int base_col = k / C;
    int extra_col = k % C;

    // 创建线程数组和对应的块信息数组
    pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
    block_thread_info *btinfo = malloc(num_threads * sizeof(block_thread_info));
    if (!threads || !btinfo)
    {
        fprintf(stderr, "线程数组分配失败！\n");
        return 1;
    }

    // 记录开始时间
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    int thread_count = 0;
    // 遍历块划分的网格，行方向 i, 列方向 j
    for (int i = 0; i < R; i++)
    {
        int row_start = i * base_row + (i < extra_row ? i : extra_row);
        int row_end = row_start + base_row + (i < extra_row ? 1 : 0);
        for (int j = 0; j < C; j++)
        {
            if (thread_count >= num_threads)
                break; // 若分块数超出线程数，退出
            int col_start = j * base_col + (j < extra_col ? j : extra_col);
            int col_end = col_start + base_col + (j < extra_col ? 1 : 0);
            btinfo[thread_count].info = &info;
            btinfo[thread_count].row_start = row_start;
            btinfo[thread_count].row_end = row_end;
            btinfo[thread_count].col_start = col_start;
            btinfo[thread_count].col_end = col_end;
            pthread_create(&threads[thread_count], NULL, block_thread_func, (void *)&btinfo[thread_count]);
            thread_count++;
        }
    }

    // 等待所有线程完成工作
    for (int i = 0; i < thread_count; i++)
    {
        pthread_join(threads[i], NULL);
    }

    // 记录结束时间
    gettimeofday(&end_time, NULL);
    double time_taken = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec) / 1e6;
    printf("矩阵相乘耗时: %.5f s\n", time_taken);

    // 清理资源
    free(info.A);
    free(info.B);
    free(info.C);
    free(threads);
    free(btinfo);

    return 0;
}
