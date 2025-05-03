#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>
#include <math.h>
#include <time.h>

// 流水线参数
int M;           // 方程总数
int num_threads; // 总线程数
int t1, t2, t3;  // 三个阶段的线程数

// 任务结构体
typedef struct
{
    double a, b, c;          // 随机系数
    double b2, fourac, twoa; // 中间量 b²、4ac、2a
    double d;                // 判别式 d = b² - 4ac
} Task;

// 环形队列结构
typedef struct
{
    Task **buf;               // 存放 Task* 的数组
    int head, tail, cap;      // 头尾索引和容量
    pthread_mutex_t m;        // 互斥锁
    pthread_cond_t not_empty; // 非空条件
    pthread_cond_t not_full;  // 非满条件
} RingQueue;

static RingQueue Q1, Q2, Q3;

// 初始化环形队列
void rq_init(RingQueue *q, int capacity)
{
    q->buf = malloc(sizeof(Task *) * capacity);
    q->cap = capacity;
    q->head = q->tail = 0;
    pthread_mutex_init(&q->m, NULL);
    pthread_cond_init(&q->not_empty, NULL);
    pthread_cond_init(&q->not_full, NULL);
}

// 销毁环形队列
void rq_destroy(RingQueue *q)
{
    free(q->buf);
    pthread_mutex_destroy(&q->m);
    pthread_cond_destroy(&q->not_empty);
    pthread_cond_destroy(&q->not_full);
}

// 向队列尾部推入元素（或 NULL 作为结束标志）
void rq_push(RingQueue *q, Task *t)
{
    pthread_mutex_lock(&q->m);
    while ((q->tail + 1) % q->cap == q->head)
        pthread_cond_wait(&q->not_full, &q->m);
    q->buf[q->tail] = t;
    q->tail = (q->tail + 1) % q->cap;
    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->m);
}

// 从队列头部弹出元素（阻塞直到非空）
Task *rq_pop(RingQueue *q)
{
    pthread_mutex_lock(&q->m);
    while (q->head == q->tail)
        pthread_cond_wait(&q->not_empty, &q->m);
    Task *t = q->buf[q->head];
    q->head = (q->head + 1) % q->cap;
    pthread_cond_signal(&q->not_full);
    pthread_mutex_unlock(&q->m);
    return t;
}

// 流水线各阶段线程函数

// Stage1：计算 b², 4ac, 2a
void *stage1(void *arg)
{
    while (1)
    {
        Task *t = rq_pop(&Q1);
        if (t == NULL)
            break;
        t->b2 = t->b * t->b;
        t->fourac = 4 * t->a * t->c;
        t->twoa = 2 * t->a;
        rq_push(&Q2, t);
    }
    return NULL;
}

// Stage2：计算判别式 d = b² - 4ac
void *stage2(void *arg)
{
    while (1)
    {
        Task *t = rq_pop(&Q2);
        if (t == NULL)
            break;
        t->d = t->b2 - t->fourac;
        rq_push(&Q3, t);
    }
    return NULL;
}

// Stage3：计算根并打印
void *stage3(void *arg)
{
    while (1)
    {
        Task *t = rq_pop(&Q3);
        if (t == NULL)
            break;
        if (t->d >= 0)
        {
            double sd = sqrt(t->d);
            double x1 = (-t->b + sd) / t->twoa;
            double x2 = (-t->b - sd) / t->twoa;
            printf("Eq(%.2f,%.2f,%.2f) → x1=%.5f, x2=%.5f\n",
                   t->a, t->b, t->c, x1, x2);
        }
        else
        {
            printf("Eq(%.2f,%.2f,%.2f) → 无实根\n",
                   t->a, t->b, t->c);
        }
        free(t);
    }
    return NULL;
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s <M> <num_threads>\n", argv[0]);
        return 1;
    }
    M = atoi(argv[1]);
    num_threads = atoi(argv[2]);
    if (M <= 0 || num_threads < 3)
    {
        fprintf(stderr, "要求 M>0 且 num_threads>=3\n");
        return 1;
    }

    // 线程按 1/3 划分到各阶段
    t1 = num_threads / 3;
    t2 = num_threads / 3;
    t3 = num_threads - t1 - t2;

    // 初始化队列
    int cap = M + num_threads + 1;
    rq_init(&Q1, cap);
    rq_init(&Q2, cap);
    rq_init(&Q3, cap);

    // 创建各阶段线程
    pthread_t *ths1 = malloc(sizeof(pthread_t) * t1);
    pthread_t *ths2 = malloc(sizeof(pthread_t) * t2);
    pthread_t *ths3 = malloc(sizeof(pthread_t) * t3);
    for (int i = 0; i < t1; i++)
        pthread_create(&ths1[i], NULL, stage1, NULL);
    for (int i = 0; i < t2; i++)
        pthread_create(&ths2[i], NULL, stage2, NULL);
    for (int i = 0; i < t3; i++)
        pthread_create(&ths3[i], NULL, stage3, NULL);

    // 随机数种子
    srand(time(NULL));
    // 推入 M 个随机 Task 到 Q1
    for (int i = 0; i < M; i++)
    {
        Task *t = malloc(sizeof(Task));
        t->a = (rand() / (double)RAND_MAX) * 200 - 100;
        t->b = (rand() / (double)RAND_MAX) * 200 - 100;
        t->c = (rand() / (double)RAND_MAX) * 200 - 100;
        rq_push(&Q1, t);
    }

    // 计时开始
    struct timeval st, ed;
    gettimeofday(&st, NULL);

    // 发送结束标志到各阶段
    for (int i = 0; i < t1; i++)
        rq_push(&Q1, NULL);
    for (int i = 0; i < t1; i++)
        pthread_join(ths1[i], NULL);

    for (int i = 0; i < t2; i++)
        rq_push(&Q2, NULL);
    for (int i = 0; i < t2; i++)
        pthread_join(ths2[i], NULL);

    for (int i = 0; i < t3; i++)
        rq_push(&Q3, NULL);
    for (int i = 0; i < t3; i++)
        pthread_join(ths3[i], NULL);

    // 计时结束并输出
    gettimeofday(&ed, NULL);
    double elapsed = (ed.tv_sec - st.tv_sec) + (ed.tv_usec - st.tv_usec) / 1e6;
    printf("使用 %d 线程处理 %d 个方程，耗时 %.5f 秒\n",
           num_threads, M, elapsed);

    // 清理
    rq_destroy(&Q1);
    rq_destroy(&Q2);
    rq_destroy(&Q3);
    free(ths1);
    free(ths2);
    free(ths3);
    return 0;
}
