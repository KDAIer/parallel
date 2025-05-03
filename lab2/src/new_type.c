#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <stddef.h>

// 使用灵活数组成员存储行数据
typedef struct
{
    int row_id;    // 行号
    double data[]; // 行数据，大小为 n（灵活数组成员，必须位于结构体最后）
} MatrixRow;

int main(int argc, char *argv[])
{
    int rank, size, m, n, k;
    double *A = NULL, *B = NULL, *C = NULL;
    MatrixRow *local_rows = NULL; // 每个进程接收的行数据（连续内存块）
    int local_row_count;          // 当前进程分到的行数
    double start, end;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
    {
        if (argc != 4)
        {
            fprintf(stderr, "Usage: %s m n k\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        k = atoi(argv[3]);

        // 分配并初始化矩阵 A、B、C
        A = (double *)malloc(m * n * sizeof(double));
        B = (double *)malloc(n * k * sizeof(double));
        C = (double *)malloc(m * k * sizeof(double));
        srand(time(NULL));
        for (int i = 0; i < m * n; i++)
            A[i] = (double)rand() / RAND_MAX;
        for (int i = 0; i < n * k; i++)
            B[i] = (double)rand() / RAND_MAX;
    }

    // 广播矩阵维度（m, n, k）
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* 计算每个进程分得的行数 */
    int base = m / size;
    int rem = m % size;
    if (rank < rem)
        local_row_count = base + 1;
    else
        local_row_count = base;

    /*
     * 创建描述单行数据的MPI数据类型。
     * 每行结构体包含：一个 int (row_id) 和 n 个 double (data)。
     */
    MPI_Datatype mpi_row_type, mpi_row_type_resized;
    int blocklengths[2] = {1, n};
    MPI_Datatype types[2] = {MPI_INT, MPI_DOUBLE};
    MPI_Aint displacements[2];

    /* 利用一个 dummy 实例计算偏移量：
       - row_id 的偏移量为 0
       - data 数组紧跟 row_id，其偏移量为 sizeof(int)
    */
    displacements[0] = 0;
    displacements[1] = sizeof(int);

    MPI_Type_create_struct(2, blocklengths, displacements, types, &mpi_row_type);
    /* 调整数据类型的 extent，使得每一行的整体内存大小为 sizeof(int) + n*sizeof(double) */
    MPI_Type_create_resized(mpi_row_type, 0, sizeof(int) + n * sizeof(double), &mpi_row_type_resized);
    MPI_Type_commit(&mpi_row_type_resized);
    MPI_Type_free(&mpi_row_type); // 不再需要原始类型

    /* 0 号进程将矩阵 A 按行打包成 m 个 MatrixRow 格式存放在 sendbuf 中 */
    MatrixRow *sendbuf = NULL;
    if (rank == 0)
    {
        sendbuf = (MatrixRow *)malloc(m * (sizeof(int) + n * sizeof(double)));
        for (int i = 0; i < m; i++)
        {
            // 每一行占用 (sizeof(int) + n*sizeof(double)) 字节
            MatrixRow *row_ptr = (MatrixRow *)((char *)sendbuf + i * (sizeof(int) + n * sizeof(double)));
            row_ptr->row_id = i;
            for (int j = 0; j < n; j++)
            {
                row_ptr->data[j] = A[i * n + j];
            }
        }
    }

    /* 为 MPI_Scatterv 准备发送计数和偏移数组（单位：自定义类型的个数，即行数） */
    int *sendcounts = NULL;
    int *displs = NULL;
    if (rank == 0)
    {
        sendcounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));
        int offset = 0;
        for (int i = 0; i < size; i++)
        {
            int rows_for_proc = (i < rem) ? (base + 1) : base;
            sendcounts[i] = rows_for_proc;
            displs[i] = offset;
            offset += rows_for_proc;
        }
    }

    /* 为接收数据分配缓冲区：local_rows 为 local_row_count 行 */
    local_rows = (MatrixRow *)malloc(local_row_count * (sizeof(int) + n * sizeof(double)));

    /* 使用 MPI_Scatterv 分发 A 的行数据 */
    MPI_Scatterv(sendbuf, sendcounts, displs, mpi_row_type_resized,
                 local_rows, local_row_count, mpi_row_type_resized,
                 0, MPI_COMM_WORLD);

    /* 广播矩阵 B 到所有进程 */
    if (rank != 0)
        B = (double *)malloc(n * k * sizeof(double));
    MPI_Bcast(B, n * k, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* 每个进程进行局部矩阵乘法，计算 local_rows 的数据乘以 B 得到局部结果 */
    double *local_C = (double *)malloc(local_row_count * k * sizeof(double));
    start = MPI_Wtime();
    for (int i = 0; i < local_row_count; i++)
    {
        MatrixRow *row = (MatrixRow *)((char *)local_rows + i * (sizeof(int) + n * sizeof(double)));
        for (int j = 0; j < k; j++)
        {
            double sum = 0.0;
            for (int l = 0; l < n; l++)
            {
                sum += row->data[l] * B[l * k + j];
            }
            local_C[i * k + j] = sum;
        }
    }
    end = MPI_Wtime();

    /* 收集所有进程的计算结果到全局矩阵 C
       使用 MPI_Gatherv 收集，每个进程贡献 local_row_count 行，每行 k 个 double */
    int *recvcounts = NULL;
    int *rdispls = NULL;
    if (rank == 0)
    {
        recvcounts = (int *)malloc(size * sizeof(int));
        rdispls = (int *)malloc(size * sizeof(int));
        int offset = 0;
        for (int i = 0; i < size; i++)
        {
            int rows_for_proc = (i < rem) ? (base + 1) : base;
            recvcounts[i] = rows_for_proc * k;
            rdispls[i] = offset;
            offset += rows_for_proc * k;
        }
    }

    if (rank == 0)
    {
        MPI_Gatherv(local_C, local_row_count * k, MPI_DOUBLE,
                    C, recvcounts, rdispls, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Gatherv(local_C, local_row_count * k, MPI_DOUBLE,
                    NULL, NULL, NULL, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    if (rank == 0)
    {
        printf("时间消耗: %f秒\n", end - start);
        // 如有需要，可打印部分结果 C
    }

    /* 释放资源 */
    if (sendbuf)
        free(sendbuf);
    free(local_rows);
    free(local_C);
    if (rank == 0)
    {
        free(A);
        free(B);
        free(C);
        free(sendcounts);
        free(displs);
        free(recvcounts);
        free(rdispls);
    }
    else
    {
        free(B);
    }
    MPI_Type_free(&mpi_row_type_resized);
    MPI_Finalize();
    return 0;
}
