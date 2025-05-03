#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <time.h>

#define IDX(i, j, n) ((i) * (n) + (j))

// 局部矩阵乘法：C += A * B
void local_matrix_multiply(double *A, double *B, double *C, int block_size)
{
    for (int i = 0; i < block_size; i++)
        for (int j = 0; j < block_size; j++)
            for (int k = 0; k < block_size; k++)
                C[IDX(i, j, block_size)] += A[IDX(i, k, block_size)] * B[IDX(k, j, block_size)];
}

int main(int argc, char *argv[])
{
    int rank, size;
    int m, n, k;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2)
    {
        if (rank == 0)
            printf("用法: %s <矩阵大小>\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    m = n = k = atoi(argv[1]);

    int q = (int)sqrt(size);
    if (q * q != size)
    {
        if (rank == 0)
            printf("进程数必须是完全平方数！\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int dims[2] = {q, q};
    int periods[2] = {1, 1}; // 开启周期性通信
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);

    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);
    int row = coords[0], col = coords[1];

    int block_size = m / q;
    double *A = NULL, *B = NULL, *C = NULL;

    double *local_A = (double *)malloc(block_size * block_size * sizeof(double));
    double *local_B = (double *)malloc(block_size * block_size * sizeof(double));
    double *local_C = (double *)calloc(block_size * block_size, sizeof(double));

    if (rank == 0)
    {
        A = (double *)malloc(m * n * sizeof(double));
        B = (double *)malloc(n * k * sizeof(double));
        C = (double *)malloc(m * k * sizeof(double));
        srand(time(NULL));
        for (int i = 0; i < m * n; i++)
            A[i] = (double)rand() / RAND_MAX;
        for (int i = 0; i < n * k; i++)
            B[i] = (double)rand() / RAND_MAX;
    }

    // 创建子矩阵类型（用于Scatterv/Gatherv）
    MPI_Datatype block_type, resized_block_type;
    MPI_Type_vector(block_size, block_size, n, MPI_DOUBLE, &block_type);
    MPI_Type_create_resized(block_type, 0, sizeof(double), &resized_block_type);
    MPI_Type_commit(&resized_block_type);

    int *sendcounts = NULL, *displs = NULL;
    if (rank == 0)
    {
        sendcounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));
        for (int i = 0; i < q; i++)
        {
            for (int j = 0; j < q; j++)
            {
                sendcounts[i * q + j] = 1;
                displs[i * q + j] = i * block_size * n + j * block_size;
            }
        }
    }

    // 分发矩阵块
    MPI_Scatterv(A, sendcounts, displs, resized_block_type, local_A,
                 block_size * block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(B, sendcounts, displs, resized_block_type, local_B,
                 block_size * block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 初始对齐：A 左移 row 次，B 上移 col 次
    int a_src, a_dst, b_src, b_dst;
    MPI_Cart_shift(cart_comm, 1, -row, &a_src, &a_dst);
    MPI_Sendrecv_replace(local_A, block_size * block_size, MPI_DOUBLE,
                         a_dst, 0, a_src, 0, cart_comm, MPI_STATUS_IGNORE);

    MPI_Cart_shift(cart_comm, 0, -col, &b_src, &b_dst);
    MPI_Sendrecv_replace(local_B, block_size * block_size, MPI_DOUBLE,
                         b_dst, 0, b_src, 0, cart_comm, MPI_STATUS_IGNORE);

    // Cannon 核心计算
    double start = MPI_Wtime();
    for (int step = 0; step < q; step++)
    {
        local_matrix_multiply(local_A, local_B, local_C, block_size);

        // A 左移一位
        MPI_Cart_shift(cart_comm, 1, -1, &a_src, &a_dst);
        MPI_Sendrecv_replace(local_A, block_size * block_size, MPI_DOUBLE,
                             a_dst, 0, a_src, 0, cart_comm, MPI_STATUS_IGNORE);

        // B 上移一位
        MPI_Cart_shift(cart_comm, 0, -1, &b_src, &b_dst);
        MPI_Sendrecv_replace(local_B, block_size * block_size, MPI_DOUBLE,
                             b_dst, 0, b_src, 0, cart_comm, MPI_STATUS_IGNORE);
    }
    double end = MPI_Wtime();

    // 收集计算结果
    MPI_Gatherv(local_C, block_size * block_size, MPI_DOUBLE,
                C, sendcounts, displs, resized_block_type,
                0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("Cannon算法完成！矩阵维度: %d x %d x %d\n", m, n, k);
        printf("总耗时: %.6f 秒\n", end - start);
    }

    free(local_A);
    free(local_B);
    free(local_C);
    if (rank == 0)
    {
        free(A);
        free(B);
        free(C);
        free(sendcounts);
        free(displs);
    }
    MPI_Type_free(&resized_block_type);
    MPI_Finalize();
    return 0;
}
