#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>
#include <time.h>

// 共享结构体
typedef struct
{
    int n;
    int *num;
} share_info;

// 每个线程的私有信息（不再存储局部和）
typedef struct
{
    share_info *info;
    int start_index;
    int end_index;
} thread_info;

// 全局累加变量及对应的互斥锁
long long global_sum = 0;
pthread_mutex_t sum_mutex = PTHREAD_MUTEX_INITIALIZER;

// 线程函数：计算区间内局部和后更新全局和
void *thread_sum(void *arg)
{
    thread_info *tinfo = (thread_info *)arg;
    share_info *info = tinfo->info;
    long long local_sum = 0;
    for (int i = tinfo->start_index; i < tinfo->end_index; i++)
    {
        local_sum += info->num[i];
    }
    // 使用互斥锁更新全局累加和
    pthread_mutex_lock(&sum_mutex);
    global_sum += local_sum;
    pthread_mutex_unlock(&sum_mutex);
    return NULL;
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        fprintf(stderr, "用法: %s <n> <num_threads>\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]);
    int num_threads = atoi(argv[2]);

    share_info info;
    info.n = n;
    info.num = (int *)malloc(n * sizeof(int));
    if (!info.num)
    {
        fprintf(stderr, "内存分配失败！\n");
        return 1;
    }

    srand((unsigned int)time(NULL));
    // 随机初始化数组
    for (int i = 0; i < n; i++)
    {
        info.num[i] = rand() % 10;
    }

    pthread_t threads[num_threads];
    thread_info tinfo[num_threads];
    int chunk_size = n / num_threads;
    int remainder = n % num_threads;
    int start_index = 0;

    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    // 创建线程并划分任务，每个线程直接在执行完局部求和后将结果累加到全局变量中
    for (int i = 0; i < num_threads; i++)
    {
        tinfo[i].info = &info;
        tinfo[i].start_index = start_index;
        if (i < remainder)
            tinfo[i].end_index = start_index + chunk_size + 1;
        else
            tinfo[i].end_index = start_index + chunk_size;
        start_index = tinfo[i].end_index;
        pthread_create(&threads[i], NULL, thread_sum, (void *)&tinfo[i]);
    }

    // 等待所有线程结束
    for (int i = 0; i < num_threads; i++)
    {
        pthread_join(threads[i], NULL);
    }

    // 记录结束时间并计算耗时
    gettimeofday(&end_time, NULL);
    double time_taken = (end_time.tv_sec - start_time.tv_sec) +
                        (end_time.tv_usec - start_time.tv_usec) / 1e6;
    printf("线程总耗时: %.5f秒\n", time_taken);
    printf("数组总和: %lld\n", global_sum);

    free(info.num);
    return 0;
}
