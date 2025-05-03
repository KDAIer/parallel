#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>

int main(int argc, char *argv[])
{
    int m, n, k;
    int rank, size;
    double *A = NULL, *B = NULL, *C = NULL;
    double *local_A = NULL, *local_C = NULL;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
    { // 主进程负责初始化矩阵
        if (argc != 4)
        {
            fprintf(stderr, "Usage: %s m n k\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        k = atoi(argv[3]);
        A = (double *)malloc(m * n * sizeof(double));
        B = (double *)malloc(n * k * sizeof(double));
        C = (double *)malloc(m * k * sizeof(double));
        srand(time(NULL));
        for (int i = 0; i < m * n; i++)
            A[i] = (double)rand() / RAND_MAX;
        for (int i = 0; i < n * k; i++)
            B[i] = (double)rand() / RAND_MAX;
    }
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD); // 广播矩阵维度信息
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD); // 广播矩阵维度信息

    // 处理A
    // 对A进行行划分
    int rows = m / size;
    local_A = (double *)malloc(rows * n * sizeof(double));
    local_C = (double *)malloc(rows * k * sizeof(double));
    // 对A分块
    MPI_Scatter(A, rows * n, MPI_DOUBLE, local_A, rows * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 处理B
    // 广播整个矩阵B
    if (rank != 0)
    {
        B = (double *)malloc(n * k * sizeof(double));
    }
    MPI_Bcast(B, n * k, MPI_DOUBLE, 0, MPI_COMM_WORLD); // 广播整个矩阵B

    // 处理局部C
    MPI_Barrier(MPI_COMM_WORLD);
    double s_time, e_time;
    s_time = MPI_Wtime();
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            local_C[i * k + j] = 0.0;
            for (int l = 0; l < n; ++l)
            {
                local_C[i * k + j] += local_A[i * n + l] * B[l * k + j];
            }
        }
    }
    // 传回C
    MPI_Gather(local_C, rows * k, MPI_DOUBLE, C, rows * k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    e_time = MPI_Wtime();

    if (rank == 0)
    {
        printf("集合通信结果\n");
        printf("矩阵大小: %d x %d x %d\n", m, n, k);
        printf("时间消耗: %f seconds\n", e_time - s_time);
    }
    // 释放内存
    free(local_A);
    free(local_C);
    if (rank == 0)
    {
        free(A);
        free(B);
        free(C);
    }
    else
    {
        free(B);
    }
    MPI_Finalize(); // 结束MPI环境
    return 0;
}