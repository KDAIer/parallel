#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>
#include <math.h>
#include <time.h>

#define THREAD_NUM 5 // 线程数：0,1,2计算中间值；3计算判别式；4计算根

// 全局系数及中间变量
double a, b, c;          // 一元二次方程的系数
double b_2, _4ac, two_a; // 分别保存 b^2, 4ac 和 2a 的值
double d;                // 判别式：d = b^2 - 4ac
double sqrtd;            // 当 d>=0 时的 sqrt(d)
double x1, x2;           // 方程的两个根

// 全局计数器用于同步：计数器达到3表示 b^2、4ac 和 2a 均已计算完成；达到4表示判别式已计算完成
int counter = 0;

// 互斥锁与条件变量，用于线程间同步
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

// 线程函数，根据线程ID执行不同的任务
void *thread_func(void *arg)
{
    int id = *(int *)arg; // 线程ID

    if (id == 0)
    {
        // 线程0计算 b^2
        double temp = b * b;
        pthread_mutex_lock(&mutex);
        b_2 = temp;
        counter++; // 完成一项计算
        printf("Thread %d: computed b^2 = %.2f. Counter = %d.\n", id, b_2, counter);
        if (counter == 3)
        {
            printf("Thread %d: All three intermediate values computed. Broadcasting condition.\n", id);
            pthread_cond_broadcast(&cond); // 唤醒等待的线程
        }
        pthread_mutex_unlock(&mutex);
    }
    else if (id == 1)
    {
        // 线程1计算 4ac
        double temp = 4 * a * c;
        pthread_mutex_lock(&mutex);
        _4ac = temp;
        counter++;
        printf("Thread %d: computed 4ac = %.2f. Counter = %d.\n", id, _4ac, counter);
        if (counter == 3)
        {
            printf("Thread %d: All three intermediate values computed. Broadcasting condition.\n", id);
            pthread_cond_broadcast(&cond);
        }
        pthread_mutex_unlock(&mutex);
    }
    else if (id == 2)
    {
        // 线程2计算 2a
        double temp = 2 * a;
        pthread_mutex_lock(&mutex);
        two_a = temp;
        counter++;
        printf("Thread %d: computed 2a = %.2f. Counter = %d.\n", id, two_a, counter);
        if (counter == 3)
        {
            printf("Thread %d: All three intermediate values computed. Broadcasting condition.\n", id);
            pthread_cond_broadcast(&cond);
        }
        pthread_mutex_unlock(&mutex);
    }
    else if (id == 3)
    {
        // 线程3等待前三个计算完成，然后计算判别式 d
        pthread_mutex_lock(&mutex);
        while (counter < 3)
        {
            printf("Thread %d: Waiting for intermediate values to be computed. Counter = %d.\n", id, counter);
            pthread_cond_wait(&cond, &mutex);
        }
        pthread_mutex_unlock(&mutex);
        d = b_2 - _4ac;
        printf("Thread %d: computed d = %.2f.\n", id, d);

        // 更新计数器，表示判别式已计算完成
        pthread_mutex_lock(&mutex);
        counter++;
        printf("Thread %d: Incremented counter after computing d. Counter = %d.\n", id, counter);
        if (counter == 4)
        {
            printf("Thread %d: d computed. Broadcasting condition to wake up thread 4.\n", id);
            pthread_cond_broadcast(&cond);
        }
        pthread_mutex_unlock(&mutex);
    }
    else if (id == 4)
    {
        // 线程4等待判别式 d 计算完成，然后计算并输出方程的根
        pthread_mutex_lock(&mutex);
        while (counter < 4)
        {
            printf("Thread %d: Waiting for d to be computed. Counter = %d.\n", id, counter);
            pthread_cond_wait(&cond, &mutex);
        }
        pthread_mutex_unlock(&mutex);
        if (d >= 0)
        {
            sqrtd = sqrt(d);
            x1 = (-b + sqrtd) / two_a;
            x2 = (-b - sqrtd) / two_a;
            printf("Thread %d: computed sqrt(d) = %.2f, x1 = %.2f, x2 = %.2f.\n", id, sqrtd, x1, x2);
        }
        else
        {
            printf("Thread %d: d < 0, no real roots.\n", id);
        }
    }
    return NULL;
}

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        printf("Usage: %s <a> <b> <c>\n", argv[0]);
        return 1;
    }
    a = atof(argv[1]);
    b = atof(argv[2]);
    c = atof(argv[3]);
    if (a == 0)
    {
        printf("Error: a cannot be zero.\n");
        return 1;
    }

    struct timeval start, end;
    gettimeofday(&start, NULL);

    pthread_t threads[THREAD_NUM];
    int thread_ids[THREAD_NUM];

    // 创建5个线程
    for (int i = 0; i < THREAD_NUM; i++)
    {
        thread_ids[i] = i;
        if (pthread_create(&threads[i], NULL, thread_func, (void *)&thread_ids[i]) != 0)
        {
            fprintf(stderr, "Error creating thread %d\n", i);
            return 1;
        }
    }

    // 等待所有线程结束
    for (int i = 0; i < THREAD_NUM; i++)
    {
        pthread_join(threads[i], NULL);
    }

    gettimeofday(&end, NULL);
    double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
    printf("Total computation time: %.5f s\n", time_taken);

    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&cond);

    return 0;
}
