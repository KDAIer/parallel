#include "parallel_for.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

typedef struct
{
    int n, k;
    float *A, *B, *C;
} matrix_args;

// 签名改为 void
void matrix_mult(int idx, void *arg)
{
    matrix_args *args = (matrix_args *)arg;
    int n = args->n;
    int k = args->k;
    // 对第 idx 行进行矩阵乘
    for (int j = 0; j < k; ++j)
    {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i)
        {
            sum += args->A[idx * n + i] * args->B[i * k + j];
        }
        args->C[idx * k + j] = sum;
    }
}

static double get_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char **argv)
{
    if (argc != 7)
    {
        fprintf(stderr,
                "Usage: %s <schedule:0=static,1=dynamic,2=guided> "
                "<num_threads> <chunk_size> <m> <n> <k>\n",
                argv[0]);
        return EXIT_FAILURE;
    }
    int schedule = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    int chunk_size = atoi(argv[3]);
    int m = atoi(argv[4]);
    int n = atoi(argv[5]);
    int k = atoi(argv[6]);

    long total = (long)m;
    float *A = malloc(sizeof(float) * m * n);
    float *B = malloc(sizeof(float) * n * k);
    float *C = malloc(sizeof(float) * m * k);
    if (!A || !B || !C)
    {
        perror("malloc");
        return EXIT_FAILURE;
    }

    // 随机初始化
    for (long i = 0; i < (long)m * n; ++i)
        A[i] = (float)rand() / RAND_MAX;
    for (long i = 0; i < (long)n * k; ++i)
        B[i] = (float)rand() / RAND_MAX;

    matrix_args args = {n, k, A, B, C};

    // 单线程静态基准
    double t0 = get_time();
    parallel_for(0, m, 1, matrix_mult, &args,
                 1, SCHEDULE_STATIC, 0);
    double t_base = get_time() - t0;

    // 并行
    double t1 = get_time();
    parallel_for(0, m, 1, matrix_mult, &args,
                 num_threads, (enum schedule_type)schedule, chunk_size);
    double t_cur = get_time() - t1;

    double speedup = t_base / t_cur;
    // 输出：schedule threads chunk m n k time_s speedup
    printf("%d %d %d %d %d %d %.6f %.2f\n",
           schedule, num_threads, chunk_size,
           m, n, k, t_cur, speedup);

    free(A);
    free(B);
    free(C);
    return EXIT_SUCCESS;
}
