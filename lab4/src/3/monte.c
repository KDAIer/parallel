#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>

int n;           // 总采样点数
int num_threads; // 线程数

long long global_count = 0; // 落在内切圆内的总点数
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

void *thread_func(void *arg)
{
    int tid = *(int *)arg;
    unsigned int seed = time(NULL) ^ tid; // 根据线程ID生成各自种子
    long long local_count = 0;

    // 每个线程计算 points_per_thread 个点
    int points_per_thread = n / num_threads;

    for (int i = 0; i < points_per_thread; i++)
    {
        double x = (double)rand_r(&seed) / RAND_MAX;
        double y = (double)rand_r(&seed) / RAND_MAX;
        // 点落在单位圆内：x^2 + y^2 <= 1
        if (x * x + y * y <= 1.0)
            local_count++;
    }
    // 若 n 不能被 num_threads 整除，则最后一个线程负责补充余数部分
    if (tid == num_threads - 1)
    {
        int remainder = n % num_threads;
        for (int i = 0; i < remainder; i++)
        {
            double x = (double)rand_r(&seed) / RAND_MAX;
            double y = (double)rand_r(&seed) / RAND_MAX;
            if (x * x + y * y <= 1.0)
                local_count++;
        }
    }

    // 更新全局计数器
    pthread_mutex_lock(&mutex);
    global_count += local_count;
    pthread_mutex_unlock(&mutex);

    printf("Thread %d: processed %d points, local count = %lld\n", tid, points_per_thread + (tid == num_threads - 1 ? n % num_threads : 0), local_count);
    return NULL;
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s <n> <num_threads>\n", argv[0]);
        return 1;
    }

    n = atoi(argv[1]);
    num_threads = atoi(argv[2]);
    if (n < 1024 || n > 65536)
    {
        fprintf(stderr, "n must be between 1024 and 65536.\n");
        return 1;
    }
    if (num_threads < 1)
    {
        fprintf(stderr, "num_threads must be at least 1.\n");
        return 1;
    }

    pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
    int *thread_ids = malloc(num_threads * sizeof(int));
    if (threads == NULL || thread_ids == NULL)
    {
        fprintf(stderr, "Memory allocation failed.\n");
        return 1;
    }

    struct timeval start, end;
    gettimeofday(&start, NULL);

    // 创建线程
    for (int i = 0; i < num_threads; i++)
    {
        thread_ids[i] = i;
        if (pthread_create(&threads[i], NULL, thread_func, &thread_ids[i]) != 0)
        {
            fprintf(stderr, "Error creating thread %d\n", i);
            return 1;
        }
    }

    // 等待所有线程结束
    for (int i = 0; i < num_threads; i++)
    {
        pthread_join(threads[i], NULL);
    }

    gettimeofday(&end, NULL);
    double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;

    // 根据 m/n 估计 π, 公式： π ≈ 4 * (内切圆内的点数) / 总点数
    double pi_estimate = 4.0 * global_count / n;

    printf("Total points: %d\n", n);
    printf("Points inside circle: %lld\n", global_count);
    printf("Estimated pi: %f\n", pi_estimate);
    printf("Time taken: %.5f s\n", time_taken);

    free(threads);
    free(thread_ids);
    pthread_mutex_destroy(&mutex);

    return 0;
}
