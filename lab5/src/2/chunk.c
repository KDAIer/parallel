#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

// 并行矩阵乘法：比较 static 与 dynamic 两种调度方式，使用可变 chunk_size
void matmul_static(double *A, double *B, double *C,
                   int m, int n, int k, int chunk)
{
#pragma omp parallel for collapse(2) schedule(static, chunk)
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++)
        {
            double sum = 0.0;
            for (int p = 0; p < n; p++)
            {
                sum += A[i * n + p] * B[p * k + j];
            }
            C[i * k + j] = sum;
        }
    }
}

void matmul_dynamic(double *A, double *B, double *C,
                    int m, int n, int k, int chunk)
{
#pragma omp parallel for collapse(2) schedule(dynamic, chunk)
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++)
        {
            double sum = 0.0;
            for (int p = 0; p < n; p++)
            {
                sum += A[i * n + p] * B[p * k + j];
            }
            C[i * k + j] = sum;
        }
    }
}

int main(int argc, char *argv[])
{
    if (argc != 6)
    {
        fprintf(stderr, "Usage: %s <m> <n> <k> <num_threads> <chunk_size>\n", argv[0]);
        return 1;
    }
    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);
    int num_threads = atoi(argv[4]);
    int chunk = atoi(argv[5]);

    omp_set_num_threads(num_threads);
    srand((unsigned)time(NULL));

    // 分配矩阵并随机初始化
    double *A = malloc(sizeof(double) * m * n);
    double *B = malloc(sizeof(double) * n * k);
    double *C = malloc(sizeof(double) * m * k);
    for (int i = 0; i < m * n; i++)
        A[i] = rand() / (double)RAND_MAX;
    for (int i = 0; i < n * k; i++)
        B[i] = rand() / (double)RAND_MAX;

    double t0, t1;

    // static 调度
    t0 = omp_get_wtime();
    matmul_static(A, B, C, m, n, k, chunk);
    t1 = omp_get_wtime();
    printf("Static(%2d): %f\n", chunk, t1 - t0);

    // dynamic 调度
    t0 = omp_get_wtime();
    matmul_dynamic(A, B, C, m, n, k, chunk);
    t1 = omp_get_wtime();
    printf("Dynamic(%2d): %f\n", chunk, t1 - t0);

    free(A);
    free(B);
    free(C);
    return 0;
}
