#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <time.h>

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s <M>\n", argv[0]);
        return 1;
    }
    int M = atoi(argv[1]);
    if (M <= 0)
    {
        fprintf(stderr, "M must be > 0\n");
        return 1;
    }

    // 随机数种子
    srand(time(NULL));

    struct timeval st, ed;
    gettimeofday(&st, NULL);

    // 单线程依次生成并求解 M 个方程
    for (int i = 0; i < M; i++)
    {
        double a = (rand() / (double)RAND_MAX) * 200 - 100;
        double b = (rand() / (double)RAND_MAX) * 200 - 100;
        double c = (rand() / (double)RAND_MAX) * 200 - 100;
        double b2 = b * b;
        double fourac = 4 * a * c;
        double twoa = 2 * a;
        double d = b2 - fourac;
        if (d >= 0)
        {
            double sd = sqrt(d);
            double x1 = (-b + sd) / twoa;
            double x2 = (-b - sd) / twoa;
            // 不打印中间结果以免干扰耗时
        }
        // 否则略过复数根
    }

    gettimeofday(&ed, NULL);
    double elapsed = (ed.tv_sec - st.tv_sec) + (ed.tv_usec - st.tv_usec) / 1e6;
    printf("单线程运行时间 %.5f s\n", elapsed);
    return 0;
}
