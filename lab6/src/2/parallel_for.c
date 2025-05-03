#include "parallel_for.h"
#include <stdlib.h>

static void *thread_func(void *arg)
{
    thread_args_t *t = (thread_args_t *)arg;

    if (t->schedule == STATIC)
    {
        // 静态分块
        int total = t->end - t->start;
        int chunk = total / t->num_threads;
        int s = t->start + t->thread_id * chunk;
        int e = (t->thread_id == t->num_threads - 1)
                    ? t->end
                    : s + chunk;
        for (int i = s; i < e; i += t->inc)
        {
            t->functor(i, t->args);
        }
    }
    else if (t->schedule == DYNAMIC)
    {
        // 动态分块
        while (1)
        {
            pthread_mutex_lock(t->mutex);
            int idx = *(t->counter);
            if (idx >= t->end)
            {
                pthread_mutex_unlock(t->mutex);
                break;
            }
            *(t->counter) += t->chunk_size;
            pthread_mutex_unlock(t->mutex);

            int s = idx;
            int e = idx + t->chunk_size;
            if (e > t->end)
                e = t->end;
            for (int i = s; i < e; i += t->inc)
            {
                t->functor(i, t->args);
            }
        }
    }
    else
    { // GUIDED
        // 引导式分块
        while (1)
        {
            pthread_mutex_lock(t->mutex);
            int idx = *(t->counter);
            if (idx >= t->end)
            {
                pthread_mutex_unlock(t->mutex);
                break;
            }
            // 剩余任务量
            int rem = t->end - idx;
            // 计算块大小：剩余除以线程数，至少 chunk_size
            int c = rem / t->num_threads;
            if (c < t->chunk_size)
                c = t->chunk_size;
            *(t->counter) += c;
            pthread_mutex_unlock(t->mutex);

            int s = idx;
            int e = idx + c;
            if (e > t->end)
                e = t->end;
            for (int i = s; i < e; i += t->inc)
            {
                t->functor(i, t->args);
            }
        }
    }

    return NULL;
}

void parallel_for(int start, int end, int inc,
                  void (*functor)(int, void *),
                  void *args,
                  int num_threads,
                  enum schedule_type schedule,
                  int chunk_size)
{
    pthread_t *threads = malloc(sizeof(pthread_t) * num_threads);
    thread_args_t *targs = malloc(sizeof(thread_args_t) * num_threads);
    pthread_mutex_t mutex;
    int *counter = malloc(sizeof(int));

    // 初始化
    *counter = start;
    pthread_mutex_init(&mutex, NULL);

    // 填充每个线程参数
    for (int t = 0; t < num_threads; ++t)
    {
        targs[t].start = start;
        targs[t].end = end;
        targs[t].inc = inc;
        targs[t].thread_id = t;
        targs[t].num_threads = num_threads;
        targs[t].chunk_size = (chunk_size > 0 ? chunk_size : 1);
        targs[t].schedule = schedule;
        targs[t].functor = functor;
        targs[t].args = args;
        targs[t].mutex = &mutex;
        targs[t].counter = counter;
    }

    // 创建线程
    for (int t = 0; t < num_threads; ++t)
    {
        pthread_create(&threads[t], NULL, thread_func, &targs[t]);
    }
    // 等待结束
    for (int t = 0; t < num_threads; ++t)
    {
        pthread_join(threads[t], NULL);
    }

    // 清理
    pthread_mutex_destroy(&mutex);
    free(counter);
    free(targs);
    free(threads);
}
