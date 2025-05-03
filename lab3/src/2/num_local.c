#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>
#include <time.h>

// 共享结构体数组
typedef struct
{
    int n;
    int *num;
} share_info;

// 每个线程的私有信息
typedef struct
{
    share_info *info;
    int start_index;
    int end_index;
    long long local_sum;
} thread_info;

// 线程函数
void *thread_sum(void *arg)
{
    thread_info *tinfo = (thread_info *)arg;
    share_info *info = tinfo->info;
    int n = info->n;
    int *num = info->num;
    long long sum = 0;

    // 计算局部和
    for (int i = tinfo->start_index; i < tinfo->end_index; i++)
    {
        sum += info->num[i];
    }
    tinfo->local_sum = sum; // 将局部和存储到结构体中
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

    // 创建线程以及设置每个线程所计算的数组区间
    pthread_t threads[num_threads];
    thread_info tinfo[num_threads];
    int chunk_size = n / num_threads;
    int remainder = n % num_threads;
    int start_index = 0;

    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    for (int i = 0; i < num_threads; i++)
    {
        tinfo[i].info = &info;
        tinfo[i].start_index = start_index;
        if (i < remainder)
        {
            tinfo[i].end_index = start_index + chunk_size + 1; // 多余的部分分配给前几个线程
        }
        else
        {
            tinfo[i].end_index = start_index + chunk_size;
        }
        tinfo[i].local_sum = 0;
        start_index = tinfo[i].end_index;
        pthread_create(&threads[i], NULL, thread_sum, (void *)&tinfo[i]);
    }

    // 等待所有线程完成工作
    long long total_sum = 0;
    for (int i = 0; i < num_threads; i++)
    {
        pthread_join(threads[i], NULL);
        total_sum += tinfo[i].local_sum;
    }

    // 记录结束时间
    gettimeofday(&end_time, NULL); // 记录结束时间
    double time_taken = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec) / 1e6;
    printf("线程总耗时: %.5f秒\n", time_taken);

    free(info.num); // 释放内存
    return 0;
}