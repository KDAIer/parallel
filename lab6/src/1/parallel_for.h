#ifndef PARALLEL_FOR_H
#define PARALLEL_FOR_H

#include <pthread.h>

enum schedule_type
{
    STATIC,
    DYNAMIC,
    GUIDED
};

typedef struct
{
    int start;                   // 循环起始
    int end;                     // 循环结束（exclusive）
    int inc;                     // 步长
    int thread_id;               // 线程编号
    int num_threads;             // 线程总数
    int chunk_size;              // 块大小（dynamic/guided 有效）
    enum schedule_type schedule; // 调度策略

    void (*functor)(int, void *); // 循环体函数
    void *args;                   // functor 的用户参数

    pthread_mutex_t *mutex; // 保护 counter 的互斥锁
    int *counter;           // dynamic/guided 的全局索引计数器
} thread_args_t;

// parallel_for 接口
void parallel_for(int start, int end, int inc,
                  void (*functor)(int, void *), void *args,
                  int num_threads, enum schedule_type schedule, int chunk_size);

#endif // PARALLEL_FOR_H
