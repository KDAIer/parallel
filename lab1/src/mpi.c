#define min(x, y) (((x) < (y)) ? (x) : (y))
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

int main()
{
    double *A = NULL, *B = NULL, *C = NULL;
    int m, n, k, i, j;
    int size, rank;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // 主进程（rank 0）输入矩阵尺寸并初始化矩阵
    if (rank == 0)
    {
        printf("Input matrix dimensions (m, n, k): ");
        scanf("%d %d %d", &m, &n, &k);
        A = (double *)malloc(m * n * sizeof(double));
        B = (double *)malloc(n * k * sizeof(double));
        C = (double *)malloc(m * k * sizeof(double));
        srand(time(NULL));
        for (i = 0; i < m * n; i++)
            A[i] = (double)rand() / RAND_MAX;
        for (i = 0; i < n * k; i++)
            B[i] = (double)rand() / RAND_MAX;
        printf("Matrix A and B initialized.\n");
    }

    // 使用点对点通信广播矩阵尺寸
    if (rank == 0)
    {
        for (int proc = 1; proc < size; proc++)
        {
            MPI_Send(&m, 1, MPI_INT, proc, 0, MPI_COMM_WORLD);
            MPI_Send(&n, 1, MPI_INT, proc, 0, MPI_COMM_WORLD);
            MPI_Send(&k, 1, MPI_INT, proc, 0, MPI_COMM_WORLD);
        }
    }
    else
    {
        MPI_Recv(&m, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&n, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&k, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // 非主进程为B分配内存
        B = (double *)malloc(n * k * sizeof(double));
    }
    printf("Process %d received matrix dimensions: m=%d, n=%d, k=%d.\n", rank, m, n, k);

    // 使用点对点通信将矩阵B广播出去
    if (rank == 0)
    {
        for (int proc = 1; proc < size; proc++)
        {
            MPI_Send(B, n * k, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD);
            printf("Process 0 sent matrix B to process %d.\n", proc);
        }
    }
    else
    {
        MPI_Recv(B, n * k, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d received matrix B.\n", rank);
    }

    // 计算每个进程应处理的行数
    int rows = m / size;   // 每个进程至少处理的行数
    int remain = m % size; // 剩余行数
    int local_rows = (rank < remain) ? (rows + 1) : rows;
    printf("Process %d computing %d rows.\n", rank, local_rows);

    // 主进程对A按行分片分发，使用点对点通信
    // 计算各进程数据的起始行
    int start_row;
    if (rank < remain)
        start_row = rank * (rows + 1);
    else
        start_row = rank * rows + remain;

    double *local_A = (double *)malloc(local_rows * n * sizeof(double));
    double *local_C = (double *)malloc(local_rows * k * sizeof(double));

    if (rank == 0)
    {
        // 复制主进程自己应处理的部分
        for (i = 0; i < local_rows; i++)
        {
            for (j = 0; j < n; j++)
            {
                local_A[i * n + j] = A[(start_row + i) * n + j];
            }
        }
        printf("Process 0 copied its portion of matrix A (rows %d to %d).\n", start_row, start_row + local_rows - 1);

        // 将其他进程对应的部分通过MPI_Send发送
        for (int proc = 1; proc < size; proc++)
        {
            int proc_rows = (proc < remain) ? (rows + 1) : rows;
            int proc_start;
            if (proc < remain)
                proc_start = proc * (rows + 1);
            else
                proc_start = proc * rows + remain;
            MPI_Send(A + proc_start * n, proc_rows * n, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD);
            printf("Process 0 sent rows %d to %d of matrix A to process %d.\n", proc_start, proc_start + proc_rows - 1, proc);
        }
    }
    else
    {
        MPI_Recv(local_A, local_rows * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d received its portion of matrix A (rows %d to %d).\n", rank, start_row, start_row + local_rows - 1);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // 局部矩阵乘法计算：local_C = local_A * B
    for (i = 0; i < local_rows; i++)
    {
        for (j = 0; j < k; j++)
        {
            local_C[i * k + j] = 0.0;
            for (int l = 0; l < n; l++)
            {
                local_C[i * k + j] += local_A[i * n + l] * B[l * k + j];
            }
        }
    }
    printf("Process %d completed local matrix multiplication.\n", rank);

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    // 收集各进程的计算结果到主进程（rank 0）
    if (rank == 0)
    {
        // 主进程先复制自己计算的结果
        for (i = 0; i < local_rows; i++)
        {
            for (j = 0; j < k; j++)
            {
                C[(start_row + i) * k + j] = local_C[i * k + j];
            }
        }
        // 接收其他进程计算的结果
        for (int proc = 1; proc < size; proc++)
        {
            int proc_rows = (proc < remain) ? (rows + 1) : rows;
            int proc_start;
            if (proc < remain)
                proc_start = proc * (rows + 1);
            else
                proc_start = proc * rows + remain;
            MPI_Recv(C + proc_start * k, proc_rows * k, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Process 0 received computed results from process %d for rows %d to %d.\n", proc, proc_start, proc_start + proc_rows - 1);
        }
    }
    else
    {
        MPI_Send(local_C, local_rows * k, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        printf("Process %d sent its computed results to process 0.\n", rank);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double local_time = end_time - start_time;
    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("Matrix multiplication time: %lf seconds\n", max_time);
    }

    // 释放内存
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
    free(local_A);
    free(local_C);

    MPI_Finalize();
    return 0;
}
