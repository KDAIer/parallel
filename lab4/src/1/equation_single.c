#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        printf("Usage: %s <a> <b> <c>\n", argv[0]);
        return 1;
    }
    double a = atof(argv[1]);
    double b = atof(argv[2]);
    double c = atof(argv[3]);

    if (a == 0)
    {
        printf("Error: a cannot be zero.\n");
        return 1;
    }

    struct timeval start, end;
    gettimeofday(&start, NULL);

    // 单线程顺序计算各中间值和最终结果：
    // 1. 计算 b^2
    double b_2 = b * b;
    printf("b^2 = %.2f\n", b_2);

    // 2. 计算 4ac
    double _4ac = 4 * a * c;
    printf("4ac = %.2f\n", _4ac);

    // 3. 计算 2a
    double two_a = 2 * a;
    printf("2a = %.2f\n", two_a);

    // 4. 计算判别式 d = b^2 - 4ac
    double d = b_2 - _4ac;
    printf("d = %.2f\n", d);

    // 5. 根据 d 的值计算方程的根（仅计算实根）
    if (d >= 0)
    {
        double sqrtd = sqrt(d);
        double x1 = (-b + sqrtd) / two_a;
        double x2 = (-b - sqrtd) / two_a;
        printf("x1 = %.2f, x2 = %.2f\n", x1, x2);
    }
    else
    {
        printf("d < 0, no real roots.\n");
    }

    gettimeofday(&end, NULL);
    double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
    printf("Time taken: %.5f seconds\n", time_taken);

    return 0;
}
