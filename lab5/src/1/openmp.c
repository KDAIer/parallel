#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <time.h>

// 使用OpenMP实现并行通用矩阵乘法
// 随机生成m×n的矩阵A及n×k的矩阵B，并对这两个矩阵进行矩阵乘法运算，得到矩阵C.
// 调度模式（默认、静态、动态调度）

int main(int argv, char *argc[])
{
    if (argv != 6)
    {
        printf("Usage: %s <m> <n> <k> <num_threads> <schedule_type>\n", argc[0]);
        return 1;
    }
    int m = atoi(argc[1]);
    int n = atoi(argc[2]);
    int k = atoi(argc[3]);
    int num_threads = atoi(argc[4]);
    char *schedule_type = argc[5];

    double *A = (double *)malloc(m * n * sizeof(double));
    double *B = (double *)malloc(n * k * sizeof(double));
    double *C = (double *)malloc(m * k * sizeof(double));

    srand((unsigned int)time(NULL));
    for (int i = 0; i < m * n; i++)
    {
        A[i] = (double)(rand() % 100) / 10.0;
    }
    for (int i = 0; i < n * k; i++)
    {
        B[i] = (double)(rand() % 100) / 10.0;
    }

    double start_time = omp_get_wtime();

    if (strcmp(schedule_type, "static") == 0)
    {
#pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < k; j++)
            {
                C[i * k + j] = 0.0;
                for (int p = 0; p < n; p++)
                {
                    C[i * k + j] += A[i * n + p] * B[p * k + j];
                }
            }
        }
    }
    else if (strcmp(schedule_type, "dynamic") == 0)
    {
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, 1)
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < k; j++)
            {
                C[i * k + j] = 0.0;
                for (int p = 0; p < n; p++)
                {
                    C[i * k + j] += A[i * n + p] * B[p * k + j];
                }
            }
        }
    }
    else
    {
#pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < k; j++)
            {
                C[i * k + j] = 0.0;
                for (int p = 0; p < n; p++)
                {
                    C[i * k + j] += A[i * n + p] * B[p * k + j];
                }
            }
        }
    }

    double end_time = omp_get_wtime();
    printf("Time taken: %f seconds\n", end_time - start_time);

    free(A);
    free(B);
    free(C);

    return 0;
}